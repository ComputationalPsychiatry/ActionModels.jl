using Test

using ActionModels
using DataFrames
import ForwardDiff

import ReverseDiff
import Mooncake
import FiniteDifferences: central_fdm
import Enzyme: set_runtime_activity, Forward, Reverse
using ADTypes:
    AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme, AutoFiniteDifferences

using StatsPlots

@testset "inference tests" begin

    ### SETUP ###
    #Generate dataset
    data = DataFrame(
        observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
        actions = vcat(
            [0, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0, 0.5, 0.8, 1, 1.5, 1.8],
            [0, 2, 0.5, 4, 5, 3],
            [0, 0.1, 0.15, 0.2, 0.25, 0.3],
            [0, 0.2, 0.4, 0.7, 1.0, 1.1],
            [0, 2, 0.5, 4, 5, 3],
        ),
        age = vcat(
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
            repeat([20], 6),
            repeat([24], 6),
            repeat([28], 6),
        ),
        id = vcat(
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
            repeat(["Hans"], 6),
            repeat(["Georg"], 6),
            repeat(["Jørgen"], 6),
        ),
        treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
    )

    #Add a second set of actions and observations
    data.actions_2 = data.actions
    data.observations_2 = data.observations

    #Add multivariate actions
    data.actions_mv = Vector.(eachrow([data.actions data.actions_2]))

    #Define observation and action cols
    observation_cols = [:observations]
    action_cols = [:actions]
    session_cols = [:id, :treatment]

    #Create model
    action_model = ActionModel(RescorlaWagner())

    #Set prior
    prior = (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = Normal(),
    )

    #Inference parameters
    n_samples = 200
    n_chains = 2

    #Go through each supported AD type
    for AD in [
        "ForwardDiff",
        "ReverseDiff",
        "ReverseDiff Compiled",
        "Mooncake",
        "Enzyme Forward",
        "Enzyme Reverse",
        "FiniteDifferences",
    ]

        #Select appropriate AD backend
        if AD == "ForwardDiff"
            ad_type = AutoForwardDiff()
        elseif AD == "ReverseDiff"
            ad_type = AutoReverseDiff()
        elseif AD == "ReverseDiff Compiled"
            ad_type = AutoReverseDiff(; compile = true)
        elseif AD == "Mooncake"
            ad_type = AutoMooncake(; config = nothing)
        elseif AD == "Enzyme Forward"
            ad_type = AutoEnzyme(; mode = set_runtime_activity(Forward, true))
        elseif AD == "Enzyme Reverse"
            ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true))
        elseif AD == "FiniteDifferences"
            ad_type = AutoFiniteDifferences(; fdm = central_fdm(5, 1))
        end



        @testset "API tests $(AD)" begin

            #Create model
            model = create_model(
                action_model,
                prior,
                data,
                observation_cols = observation_cols,
                action_cols = action_cols,
                session_cols = session_cols,
            )

            @testset "sampling and extracting results $(AD)" begin
                #Posterior
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
                posterior_parameters = get_session_parameters!(model, :posterior)
                posterior_parameters_df = summarize(posterior_parameters)
                @test sort(posterior_parameters_df, :learning_rate).id ==
                      ["Hans", "Hans", "Georg", "Georg", "Jørgen", "Jørgen"]
                summarize(posterior_parameters, std)
                posterior_trajectories =
                    get_state_trajectories!(model, :expected_value, :posterior)
                summarize(posterior_trajectories)
                summarize(posterior_trajectories, std)

                #Prior
                prior_chains =
                    sample_prior!(model, n_samples = n_samples, n_chains = n_chains)
                prior_parameters = get_session_parameters!(model, :prior)
                summarize(prior_parameters)
                prior_trajectories = get_state_trajectories!(model, :expected_value, :prior)
                summarize(prior_trajectories)
            end

            @testset "plotting results $(AD)" begin
                ### Test plotting functions ###
                #Default Turing plots
                plot(model.posterior.chains)
                plot(model.prior.chains)

                # plot(posterior_parameters)
                # plot(posterior_parameters, session = "id:Hans.treatment:control")
                # plot(posterior_trajectories)
                # plot(posterior_trajectories, session = "id:Hans.treatment:control")

                # plot(prior_parameters)
                # plot(prior_parameters, session = "id:Hans.treatment:control")
                # plot(prior_trajectories)
                # plot(prior_trajectories, session = "id:Hans.treatment:control")
            end

            @testset "sample_posterior! variations $(AD)" begin

                @testset "different init_params $(AD)" begin
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = nothing,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :MLE,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :MAP,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                    posterior_chains = sample_posterior!(
                        model,
                        resample = true,
                        init_params = :sample_prior,
                        ad_type = ad_type,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                end

                @testset "multi-core sampling $(AD)" begin
                    using Distributed

                    addprocs(2)

                    @everywhere using ActionModels
                    @everywhere model = $model

                    posterior_chains = sample_posterior!(
                        model,
                        MCMCDistributed(),
                        resample = true,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )

                    rmprocs(workers())
                end

                @testset "multi-thread sampling $(AD)" begin
                    posterior_chains = sample_posterior!(
                        model,
                        MCMCThreads(),
                        resample = true,
                        n_samples = n_samples,
                        n_chains = n_chains,
                    )
                end

                @testset "save/resume $(AD)" begin
                    #Prepare a specific directory
                    save_resume = SampleSaveResume(
                        path = mktempdir(),
                        save_every = Int64(n_samples / 2),
                    )

                    @testset "first run" begin

                        posterior_chains = sample_posterior!(
                            model,
                            resample = true,
                            save_resume = save_resume,
                            n_samples = n_samples,
                            n_chains = n_chains,
                        )
                    end

                    @testset "continuing run" begin
                        #TODO THIS ERRORS
                        # posterior_chains = sample_posterior!(
                        #     model,
                        #     resample = true,
                        #     save_resume = save_resume,
                        #     n_samples = n_samples*2,
                        #     n_chains = n_chains,
                        # )
                    end

                    @testset "multi-core save/resume $(AD)" begin
                        using Distributed

                        addprocs(2)

                        @everywhere using ActionModels
                        @everywhere model = $model

                        posterior_chains = sample_posterior!(
                            model,
                            MCMCDistributed(),
                            resample = true,
                            save_resume = SampleSaveResume(
                                path = mktempdir(),
                                save_every = Int64(n_samples / 2),
                            ),
                            n_samples = n_samples,
                            n_chains = n_chains,
                        )
                        rmprocs(workers())
                    end

                    @testset "multi-thread save/resume $(AD)" begin
                        posterior_chains = sample_posterior!(
                            model,
                            MCMCThreads(),
                            resample = true,
                            save_resume = SampleSaveResume(
                                path = mktempdir(),
                                save_every = Int64(n_samples / 2),
                            ),
                            n_samples = n_samples,
                            n_chains = n_chains,
                        )
                    end
                end
            end
        end



        @testset "population model tests $(AD)" begin

            @testset "single session ($AD)" begin
                #Extract observations and actions from data
                observations = data[!, :observations]
                actions = data[!, :actions]

                #Create model
                model = create_model(action_model, prior, observations, actions)

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "independent sessions model ($AD)" begin

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )

                #Check that the posteriors are correct
                posterior_parameters = get_session_parameters!(model, :posterior)
                posterior_parameters_df = summarize(posterior_parameters)
                @test sort(posterior_parameters_df, :learning_rate).id ==
                      ["Hans", "Hans", "Georg", "Georg", "Jørgen", "Jørgen"]
            end

            @testset "custom population model ($AD)" begin

                #Define population model
                @model function custom_population_model()
                    #Sample parameters (same across sessions)
                    control_learning_rate ~ LogitNormal(0.0, 1.0)
                    control_action_noise ~ LogNormal(0.0, 1.0)
                    control_initial_value ~ Normal(0.0, 1.0)
                    treatment_learning_rate ~ LogitNormal(0.0, 1.0)
                    treatment_action_noise ~ LogNormal(0.0, 1.0)
                    treatment_initial_value ~ Normal(0.0, 1.0)

                    #Put into vector
                    control_params = [
                        (
                            control_learning_rate,
                            control_action_noise,
                            control_initial_value,
                        ) for _ = 1:3
                    ]
                    treatment_params = [
                        (
                            treatment_learning_rate,
                            treatment_action_noise,
                            treatment_initial_value,
                        ) for _ = 1:3
                    ]
                    #Return vector
                    return [control_params; treatment_params]
                end

                model = create_model(
                    action_model,
                    custom_population_model(),
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    parameters_to_estimate = (
                        :learning_rate,
                        :action_noise,
                        :initial_value,
                    ),
                )

                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end



        @testset "action model variations $(AD)" begin

            @testset "multivariate parameter $(AD)" begin
                function multivariate_parameter(
                    attributes::ModelAttributes,
                    observation::Float64,
                )

                    parameters = load_parameters(attributes)
                    vector_noise = parameters.vector_noise

                    return Normal(vector_noise[1], exp(vector_noise[2]))
                end

                #Create model
                new_model = ActionModel(
                    multivariate_parameter,
                    parameters = (; vector_noise = Parameter([0, 1])),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action = Action(Normal)),
                )

                #Set prior
                new_prior = (; vector_noise = filldist(Normal(0.0, 1.0), 2))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )

                summarize(get_session_parameters!(model))
            end

            @testset "multivariate state $(AD)" begin
                function multivariate_state(
                    attributes::ModelAttributes,
                    observation::Float64,
                )

                    parameters = load_parameters(attributes)
                    states = load_states(attributes)

                    noise = parameters.noise

                    vector_means = states.vector_means

                    new_vector_means =
                        [vector_means[1] + observation, vector_means[2] - observation]

                    update_state!(attributes, :vector_means, new_vector_means)

                    return Normal(sum(vector_means), noise)
                end

                #Create model
                new_model = ActionModel(
                    multivariate_state,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    states = (; vector_means = State([0.0, 0.0])),
                    actions = (; action = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )

                get_session_parameters!(model, :posterior)
                state_trajectories =
                    get_state_trajectories!(model, :vector_means, :posterior)
                summarize(state_trajectories)
            end

            @testset "multivariate action $(AD)" begin
                function multivariate_action(
                    attributes::ModelAttributes,
                    observation::Float64,
                )

                    parameters = load_parameters(attributes)

                    noise = parameters.noise

                    return filldist(Normal(observation, noise), 2)
                end

                #Create model
                new_model = ActionModel(
                    multivariate_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action = Action(Distribution{Multivariate,Continuous})),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = [:actions_mv],
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "check for parameter rejections, no rejections $(AD)" begin
                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    parameter_rejections = ParameterChecking(),
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "check for parameter rejections, with rejections $(AD)" begin

                function rejected_params(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    if observation * noise > 5.0
                        throw(RejectParameters("This parameter is rejected"))
                    end

                    return Normal(observation, noise)
                end
                #Create model
                new_model = ActionModel(
                    rejected_params,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    parameter_rejections = ParameterChecking(),
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "depend on previous action ($AD)" begin

                function dependent_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    previous_action = load_actions(attributes)
                    noise = parameters.noise

                    previous_action = previous_action.action_1

                    if ismissing(previous_action)
                        previous_action = 0.0
                    end

                    actiondist = Normal(observation + previous_action, noise)

                    return actiondist
                end
                #Create model
                new_model = ActionModel(
                    dependent_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action_1 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "depend on previous action, multiple actions ($AD)" begin

                function dependent_multi_action(
                    attributes::ModelAttributes,
                    observation::Float64,
                )

                    parameters = load_parameters(attributes)

                    noise = parameters.noise

                    previous_action = load_actions(attributes)

                    if ismissing(previous_action[1])
                        previous_action = (0.0, 0.0)
                    end

                    actiondist1 = Normal(observation + previous_action[1], noise)
                    actiondist2 = Normal(observation - previous_action[2], noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    dependent_multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end


        @testset "data context variations ($AD)" begin

            @testset "varying session lengths $(AD)" begin

                #Remove last two rows from the data
                new_data = data[1:(end-2), :]

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )

                get_session_parameters!(model, :posterior)
                state_trajectories =
                    get_state_trajectories!(model, :expected_value, :posterior)
                summarize(state_trajectories)
            end

            @testset "independent sessions model, single session column ($AD)" begin
                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = [:id],
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "independent sessions model, no session columns ($AD)" begin
                #Create model
                model = create_model(
                    action_model,
                    prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end


            @testset "multiple actions ($AD)" begin

                function multi_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple observations ($AD)" begin

                function multi_observation(
                    attributes::ModelAttributes,
                    observation1::Float64,
                    observation2::Float64,
                )

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist = Normal(observation1, noise)

                    return actiondist
                end
                #Create model
                new_model = ActionModel(
                    multi_observation,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = [:observations, :observations_2],
                    action_cols = action_cols,
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple observations and multiple actions ($AD)" begin

                function multi_observation_action(
                    attributes::ModelAttributes,
                    observation1::Float64,
                    observation2::Float64,
                )

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation1, noise)
                    actiondist2 = Normal(observation2, noise)

                    return (actiondist1, actiondist2)
                end
                #Create action_model
                new_model = ActionModel(
                    multi_observation_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    data,
                    observation_cols = [:observations, :observations_2],
                    action_cols = [:actions, :actions_2],
                    session_cols = session_cols,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "single session with multiple observations and actions ($AD)" begin
                function multi_observation_action(
                    attributes::ModelAttributes,
                    observation1::Float64,
                    observation2::Float64,
                )
                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation1, noise)
                    actiondist2 = Normal(observation2, noise)

                    return (actiondist1, actiondist2)
                end

                #Create action_model
                new_model = ActionModel(
                    multi_observation_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (;
                        observation_1 = Observation(Float64),
                        observation_2 = Observation(Float64),
                    ),
                    actions = (; action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Extract observations and actions from data
                observations = Tuple.(eachrow(data[!, [:observations, :observations_2]]))
                actions = Tuple.(eachrow(data[!, [:actions, :actions_2]]))

                #Create model
                model = create_model(new_model, new_prior, observations, actions)

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end



        @testset "missing actions $(AD)" begin

            @testset "impute missing actions ($AD)" begin

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, :actions)
                new_data[[2, 7, 12], :actions] .= missing

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    missing_actions = InferMissingActions(),
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "skip missing actions ($AD)" begin

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, :actions)
                new_data[[2, 7, 12], :actions] .= missing

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    impute_missing_actions = false,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "single session, impute missing actions" begin

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, :actions)
                new_data[[2, 7, 12], :actions] .= missing

                #Extract observations and actions from data
                observations = new_data[!, :observations]
                actions = new_data[!, :actions]

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    observations,
                    actions,
                    impute_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "single session, skip missing actions $(AD)" begin

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, :actions)
                new_data[[2, 7, 12], :actions] .= missing

                #Extract observations and actions from data
                observations = new_data[!, :observations]
                actions = new_data[!, :actions]

                #Create model
                model = create_model(
                    action_model,
                    prior,
                    observations,
                    actions,
                    impute_missing_actions = false,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple actions, impute missing actions ($AD)" begin

                function multi_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, [:actions, :actions_2])
                new_data[[2, 12], :actions] .= missing
                new_data[[3], :actions_2] .= missing

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    session_cols = session_cols,
                    impute_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "single session, impute missing actions, multiple actions ($AD)" begin

                function multi_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, [:actions, :actions_2])
                new_data[[2, 12], :actions] .= missing
                new_data[[3], :actions_2] .= missing

                #Extract observations and actions from data
                observations = new_data[!, :observations]
                actions = Tuple{Union{Missing,Float64},Union{Missing,Float64}}[
                    Tuple(row) for row in eachrow(new_data[!, [:actions, :actions_2]])
                ]
                #Create model

                model = create_model(
                    new_model,
                    new_prior,
                    observations,
                    actions,
                    impute_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
            @testset "single session, skip missing actions, multiple actions ($AD)" begin

                function multi_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, [:actions, :actions_2])
                new_data[[2, 12], :actions] .= missing
                new_data[[3], :actions_2] .= missing

                #Extract observations and actions from data
                observations = new_data[!, :observations]
                actions = Tuple{Union{Missing,Float64},Union{Missing,Float64}}[
                    Tuple(row) for row in eachrow(new_data[!, [:actions, :actions_2]])
                ]

                #Create model
                model = create_model(new_model, new_prior, observations, actions)

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "multiple actions, skip missing actions ($AD)" begin

                function multi_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    noise = parameters.noise

                    actiondist1 = Normal(observation, noise)
                    actiondist2 = Normal(observation, noise)

                    return (actiondist1, actiondist2)
                end
                #Create model
                new_model = ActionModel(
                    multi_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (action_1 = Action(Normal), action_2 = Action(Normal)),
                )

                #Set prior
                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, [:actions, :actions_2])
                new_data[[2, 12], :actions] .= missing
                new_data[[3], :actions_2] .= missing

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = [:actions, :actions_2],
                    session_cols = session_cols,
                    impute_missing_actions = false,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end

            @testset "depend on previous action, impute missing actions $(AD)" begin

                function dependent_action(attributes::ModelAttributes, observation::Float64)

                    parameters = load_parameters(attributes)
                    previous_action = load_actions(attributes)
                    noise = parameters.noise

                    previous_action = previous_action.action_1

                    if ismissing(previous_action)
                        previous_action = 0.0
                    end

                    actiondist = Normal(observation + previous_action, noise)

                    return actiondist
                end

                #Create model
                new_model = ActionModel(
                    dependent_action,
                    parameters = (; noise = Parameter(1.0)),
                    observations = (; observation = Observation(Float64)),
                    actions = (; action_1 = Action(Normal)),
                )

                new_prior = (; noise = LogNormal(0.0, 1.0))

                #Create new dataframe where three actions = missing
                new_data = allowmissing(data, [:actions, :actions_2])
                new_data[[2, 12], :actions] .= missing

                #Create model
                model = create_model(
                    new_model,
                    new_prior,
                    new_data,
                    observation_cols = observation_cols,
                    action_cols = action_cols,
                    session_cols = session_cols,
                    impute_missing_actions = true,
                )

                #Fit model
                posterior_chains = sample_posterior!(
                    model,
                    ad_type = ad_type,
                    n_samples = n_samples,
                    n_chains = n_chains,
                )
            end
        end
    end
end
