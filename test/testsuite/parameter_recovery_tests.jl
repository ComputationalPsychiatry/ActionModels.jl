using Test
using ActionModels
using Distributed

@testset "parameter recovery" begin

    @testset "non-parallelized" begin

        #Agent model to do recovery on
        agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

        #Parameters to be recovered
        parameter_ranges = Dict(
            "learning_rate" => collect(0:0.5:1),
            ("initial", "value") => collect(-2:2:2),
            "action_noise" => collect(0:1:3),
        )

        #Input sequences to use
        input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

        #Sets of priors to use
        priors = [
            Dict(
                "learning_rate" => LogitNormal(0, 1),
                ("initial", "value") => Normal(0, 1),
                "action_noise" => truncated(Normal(0, 1), lower = 0),
            ),
            Dict(
                "learning_rate" => LogitNormal(0, 0.1),
                ("initial", "value") => Normal(0, 0.1),
                "action_noise" => truncated(Normal(0, 0.1), lower = 0),
            ),
        ]

        #Times to repeat each simulation
        n_simulations = 2

        #Sampler settings
        sampler_settings = (n_iterations = 10, n_chains = 1)

        #Run parameter recovery
        results_df = parameter_recovery(
            agent,
            parameter_ranges,
            input_sequence,
            priors,
            n_simulations,
            sampler_settings = sampler_settings,
            show_progress = false,
        )

        @test results_df isa DataFrame

    end

    @testset "parallelized" begin

        addprocs(4)

        @everywhere begin
            using ActionModels

            #Agent model to do recovery on
            agent = premade_agent("continuous_rescorla_wagner_gaussian", verbose = false)

            #Parameters to be recovered
            parameter_ranges = Dict(
                "learning_rate" => collect(0:0.1:1),
                ("initial", "value") => collect(-2:1:2),
                "action_noise" => collect(0:0.5:3),
            )

            #Input sequences to use
            input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

            #Sets of priors to use
            priors = [
                Dict(
                    "learning_rate" => LogitNormal(0, 1),
                    ("initial", "value") => Normal(0, 1),
                    "action_noise" => truncated(Normal(0, 1), lower = 0),
                ),
                Dict(
                    "learning_rate" => LogitNormal(0, 0.1),
                    ("initial", "value") => Normal(0, 0.1),
                    "action_noise" => truncated(Normal(0, 0.1), lower = 0),
                ),
            ]

            #Times to repeat each simulation
            n_simulations = 2

            #Sampler settings
            sampler_settings = (n_iterations = 10, n_chains = 1)
        end

        #Run parameter recovery
        results_df = parameter_recovery(
            agent,
            parameter_ranges,
            input_sequence,
            priors,
            n_simulations,
            sampler_settings = sampler_settings,
            parallel = true,
            show_progress = false,
        )

        rmprocs(workers())

        @test results_df isa DataFrame

    end

    @testset "check for parameter rejections" begin
        
        #Action model which can error 
        function action_with_errors(agent, input::R) where {R<:Real}

            noise = agent.parameters["noise"]

            if noise > 2.9
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
        new_priors = Dict("noise" => truncated(Normal(0.0, 1.0), lower = 0, upper = 3.1))

        #Parameters to be recovered
        new_parameter_ranges = Dict(
            "noise" => collect(0:0.5:3),
        )

        #Input sequences to use
        input_sequence = [[1, 2, 1, 0, 0, 1, 1, 2, 1, 2], [2, 3, 1, 5, 4, 8, 6, 4, 5]]

        #Times to repeat each simulation
        n_simulations = 2

        #Sampler settings
        sampler_settings = (n_iterations = 1000, n_chains = 1)

        #Run parameter recovery
        results_df = parameter_recovery(
            new_agent,
            new_parameter_ranges,
            input_sequence,
            new_priors,
            n_simulations,
            sampler_settings = sampler_settings,
            show_progress = false,
            check_parameter_rejections = true,
        )

        @test results_df isa DataFrame

    end
end
