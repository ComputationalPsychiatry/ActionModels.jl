using Test
using StatsPlots
using ActionModels, DataFrames
using AxisArrays, Turing
using Turing: AutoReverseDiff


@testset "fitting tests" begin

    ### SETUP ###
    #Create dataframe
    data = DataFrame(
        inputs = repeat([1, 1, 1, 2, 2, 2], 3),
        inputs_2 = repeat([1, 1, 1, 2, 2, 2], 3),
        actions = [
            0,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0,
            0.5,
            0.8,
            1,
            1.5,
            1.8,
            0,
            2,
            0.5,
            4,
            5,
            3,
        ],
        actions_2 = [
            0,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0,
            0.5,
            0.8,
            1,
            1.5,
            1.8,
            0,
            2,
            0.5,
            4,
            5,
            3,
        ],
        age = vcat([repeat([20], 6), repeat([22], 6), repeat([28], 6)]...),
        category = vcat([repeat(["1"], 6), repeat(["2"], 6), repeat(["2"], 6)]...),
        id = vcat([repeat(["Hans"], 6), repeat(["Georg"], 6), repeat(["Jørgen"], 6)]...),
    )

    #Create agent
    agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

    #Set prior
    prior = Dict(
        "learning_rate" => LogitNormal(0.0, 1.0),
        "action_noise" => truncated(Normal(0.0, 1.0), lower = 0),
        ("initial", "value") => Normal(0.0, 1.0),
    )

    #Set samplings settings
    sampler = NUTS(-1, 0.65; adtype = AutoReverseDiff(; compile = true))
    n_iterations = 1000
    sampling_kwargs = (; progress = false)

    @testset "single agent" begin
        #Extract inputs and actions from data
        inputs = data[!, :inputs]
        actions = data[!, :actions]

        #Create model
        model = create_model(agent, prior, inputs, actions)

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        #agent_parameters = extract_quantities(model, fitted_model)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "simple statistical model" begin
        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)
        #Rename chains
        renamed_model = rename_chains(fitted_model, model)

        #Extract agent parameters
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters, DataFrame)
        estimates_dict = get_estimates(agent_parameters, Dict)
        #estimates_chains = get_estimates(agent_parameters, Chains)

        #Extract state trajectories
        state_trajectories = get_trajectories(model, fitted_model, ["value", "action"])
        trajectory_estimates_df = get_estimates(state_trajectories)

        #Check that the learning rates are estimated right
        @test estimates_df[!, :learning_rate] == sort(estimates_df[!, :learning_rate])

        @test state_trajectories isa AxisArrays.AxisArray{
            Union{Missing,Float64},
            5,
            Array{Union{Missing,Float64},5},
            Tuple{
                AxisArrays.Axis{:agent,Vector{Symbol}},
                AxisArrays.Axis{:state,Vector{Symbol}},
                AxisArrays.Axis{:timestep,UnitRange{Int64}},
                AxisArrays.Axis{:sample,UnitRange{Int64}},
                AxisArrays.Axis{:chain,UnitRange{Int64}},
            },
        }
        @test agent_parameters isa AxisArrays.AxisArray{
            Float64,
            4,
            Array{Float64,4},
            Tuple{
                AxisArrays.Axis{:agent,Vector{Symbol}},
                AxisArrays.Axis{:parameter,Vector{Symbol}},
                AxisArrays.Axis{:sample,UnitRange{Int64}},
                AxisArrays.Axis{:chain,UnitRange{Int64}},
            },
        }

        #Fit model
        prior_chains = sample(model, Prior(), n_iterations; sampling_kwargs...)
        renamed_prior_chains = rename_chains(prior_chains, model)

        plot_parameters(renamed_prior_chains, renamed_model)

        prior_trajectories = get_trajectories(model, prior_chains, ["value", "action"])
        plot_trajectories(prior_trajectories)
        plot_trajectories(state_trajectories)
    end

    @testset "custom statistical model" begin

    end

    @testset "no grouping cols" begin
        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = Symbol[],
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        # #Extract quantities
        # agent_parameters = extract_quantities(model, fitted_model)

        # #Rename chains
        # renamed_model = rename_chains(fitted_model, model)
    end

    @testset "multiple grouping cols" begin

        #Create model
        model = create_model(
            agent,
            prior,
            data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = [:id, :category],
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)
        #Rename chains
        renamed_model = rename_chains(fitted_model, model)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Extract state trajectories
        state_trajectories = get_trajectories(model, fitted_model, ["value", "action"])
        trajectory_estimates_df = get_estimates(state_trajectories)


        #Check that the learning rates are estimated right
        @test estimates_df[!, :learning_rate] == sort(estimates_df[!, :learning_rate])


    end

    @testset "missing actions" begin

        #Create new dataframe where three actions = missing
        new_data = allowmissing(data, :actions)
        new_data[[2, 7, 12], :actions] .= missing

        #Create model
        model = create_model(
            agent,
            prior,
            new_data,
            input_cols = :inputs,
            action_cols = :actions,
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "multiple actions" begin

        #Action model with multiple actions    
        function multi_action(agent, input::Real)

            noise = agent.parameters["noise"]

            actiondist1 = Normal(input, noise)
            actiondist2 = Normal(input, noise)

            return [actiondist1, actiondist2]
        end
        #Create agent
        new_agent = init_agent(multi_action, parameters = Dict("noise" => 1.0))

        #Set prior
        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = :inputs,
            action_cols = [:actions, :actions_2],
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)

    end

    @testset "multiple actions, missing actions" begin
        #Action model with multiple actions    
        function multi_action(agent, input::Real)

            noise = agent.parameters["noise"]

            actiondist1 = Normal(input, noise)
            actiondist2 = Normal(input, noise)

            return [actiondist1, actiondist2]
        end
        #Create agent
        new_agent = init_agent(multi_action, parameters = Dict("noise" => 1.0))

        #Set prior
        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create new dataframe where three actions = missing
        new_data = allowmissing(data, [:actions, :actions_2])
        new_data[[2, 12], :actions] .= missing
        new_data[[3], :actions_2] .= missing

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            new_data,
            input_cols = :inputs,
            action_cols = [:actions, :actions_2],
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "multiple inputs" begin

        #Action model with multiple actions    
        function multi_input(agent, input::Tuple{R,R}) where {R<:Real}

            noise = agent.parameters["noise"]

            input1, input2 = input

            actiondist = Normal(input1, noise)

            return actiondist
        end
        #Create agent
        new_agent = init_agent(multi_input, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = [:inputs, :inputs_2],
            action_cols = :actions,
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "multiple inputs and multiple actions" begin

        #Action model with multiple actions    
        function multi_input_action(agent, input::Tuple{R,R}) where {R<:Real}

            noise = agent.parameters["noise"]

            input1, input2 = input

            actiondist1 = Normal(input1, noise)
            actiondist2 = Normal(input2, noise)

            return [actiondist1, actiondist2]
        end
        #Create agent
        new_agent = init_agent(multi_input_action, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => LogNormal(0.0, 1.0))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = [:inputs, :inputs_2],
            action_cols = [:actions, :actions_2],
            grouping_cols = :id,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)

        #Extract quantities
        agent_parameters = extract_quantities(model, fitted_model)
        estimates_df = get_estimates(agent_parameters)

        #Rename chains
        renamed_model = rename_chains(fitted_model, model)
    end

    @testset "Check for parameter rejections" begin
        #Action model which can error
        function action_with_errors(agent, input::R) where {R<:Real}

            noise = agent.parameters["noise"]

            if noise > 2.5
                 #Throw an error that will reject samples when fitted
                throw(
                    RejectParameters(
                        "Rejected noise",
                    ),
                )
            end

            actiondist = Normal(input, noise)

            return actiondist
        end
        #Create agent
        new_agent = init_agent(action_with_errors, parameters = Dict("noise" => 1.0))

        new_prior = Dict("noise" => truncated(Normal(0.0, 1.0), lower = 0, upper = 3.1))

        #Create model
        model = create_model(
            new_agent,
            new_prior,
            data,
            input_cols = [:inputs],
            action_cols = [:actions],
            grouping_cols = :id,
            check_parameter_rejections = true,
        )

        #Fit model
        fitted_model = sample(model, sampler, n_iterations; sampling_kwargs...)
    end
end