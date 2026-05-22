using ActionModels
using Test
using Distributions

@testset "Submodel and MultiSubmodel tests" begin

    @testset "MultiSubmodel initialization and maps" begin
        sub1 = ActionModels.ContinuousRescorlaWagner(learning_rate=0.1, initial_value=0.0)
        sub2 = ActionModels.ContinuousRescorlaWagner(learning_rate=0.2, initial_value=0.5)
        
        multi = MultiSubmodel((m1=sub1, m2=sub2))
        
        # Test parameter names and types
        param_types = ActionModels.get_parameter_types(multi)
        @test keys(param_types) == (:m1_initial_value, :m1_learning_rate, :m2_initial_value, :m2_learning_rate)
        @test param_types.m1_learning_rate == Float64
        
        # Test state names and types
        state_types = ActionModels.get_state_types(multi)
        @test keys(state_types) == (:m1_expected_value, :m2_expected_value)
        @test state_types.m1_expected_value == Float64
        
        # Test maps in MultiSubmodel
        @test multi.parameter_map.m1_learning_rate == (:m1, :learning_rate)
        @test multi.state_map.m1_expected_value == (:m1, :expected_value)
    end

    @testset "ActionModel with MultiSubmodel" begin
        sub1 = ActionModels.ContinuousRescorlaWagner(learning_rate=0.1, initial_value=0.0)
        sub2 = ActionModels.ContinuousRescorlaWagner(learning_rate=0.2, initial_value=0.5)
        
        # Test ActionModel with MultiSubmodel via NamedTuple (should be converted automatically)
        model_fn = (attributes, obs) -> Normal(attributes.submodel.m1_expected_value + attributes.submodel.m2_expected_value, 1.0)
        
        model = ActionModel(
            model_fn,
            parameters = (p1 = Parameter(1.0),),
            actions = (a1 = Action(Normal),),
            submodel = (m1=sub1, m2=sub2)
        )
        
        @test model.submodel isa MultiSubmodel
        @test keys(model.submodel.submodels) == (:m1, :m2)
        
        # Test initialization of attributes
        agent = init_agent(model)
        @test agent.attributes.submodel isa ActionModels.MultiSubmodelAttributes
        
        # Test getting submodel parameters through merged names
        @test ActionModels.get_parameters(agent.attributes.submodel, :m1_learning_rate) == 0.1
        @test ActionModels.get_parameters(agent.attributes.submodel, :m2_learning_rate) == 0.2
        
        # Test setting submodel parameters
        ActionModels.set_parameters!(agent.attributes.submodel, :m1_learning_rate, 0.5)
        @test ActionModels.get_parameters(agent.attributes.submodel, :m1_learning_rate) == 0.5
        
        # Test getting states
        @test ActionModels.get_states(agent.attributes.submodel, :m1_expected_value) == 0.0
        @test ActionModels.get_states(agent.attributes.submodel, :m2_expected_value) == 0.5
    end

end
