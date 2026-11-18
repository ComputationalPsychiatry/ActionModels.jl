using Test
using ActionModels

@testset "simulation" begin
    ### SETUP ###
    action_model = ActionModel(RescorlaWagner())

    agent = init_agent(action_model, save_history = :expected_value)

    @testset "give_observations!" begin

        observations = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

        actions = simulate!(agent, observations)

        @test length(actions) == length(observations)

    end

    @testset "parameter and state API" begin
        #Variations of get_states
        get_states(agent)

        get_states(agent, :expected_value)

        #Variations of get_parameters
        get_parameters(agent)

        get_parameters(agent, :initial_value)

        get_parameters(agent, (:initial_value, :learning_rate))

        #Variations of set_parameters
        set_parameters!(agent, :initial_value, 1.0)

        #Variations of get_history
        get_history(agent, :expected_value)

        get_history(agent)

        reset!(agent)
    end

    @testset "actions from distributions of different types" begin
        #An action model with a continuous and a binary action
        function two_reports(attributes::ModelAttributes, observation::Float64)
            return (Normal(observation, 1.0), Bernoulli(0.5))
        end
        two_action_model = ActionModel(
            two_reports;
            parameters = (; noise = Parameter(1.0)),
            observations = (; observation = Observation(Float64)),
            actions = (; report = Action(Normal), choice = Action(Bernoulli)),
        )
        two_action_agent = init_agent(two_action_model)

        observations = [0.1, 0.5, 0.9]

        actions = simulate!(two_action_agent, observations)

        @test length(actions) == length(observations)
        @test all(action -> action[1] isa Float64 && action[2] isa Bool, actions)
    end
end
