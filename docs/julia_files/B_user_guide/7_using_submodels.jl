# # Using submodels

# When making complex action models, it can be useful to have a structure with it's own set of methods and dat containers that can be used inside the model.
# This can also be useful when one wants to pre-create part of the model outside of the model fitting function, for example to save time.
# In ActionModels, such a struct is called a `Submodel`.
# Due to the design of Turing, there are some requirements to submodels thst are necessary for avoiding type-instability.
# In ActionModels, this is ensured by using creating a Submodel type and passing it to the `ActionModel` constructor.
# A set of functions from the ActionModels package must then be extended to ensure that the submodel can interface smoothly with the rest of the package.
# In this section, we will demonstrate how to do this, using a simple example of a Rescorla-Wagner model, which is a simple reinforcement learning model that learns expectations about observations based on previous observations.
# This is identical to the model created in the defining models REF section, but here we will create it as a submodel.
# The Rescorla-Wagner model is simnple enough that it is not necessary to use a Submodel for creating it, but it can serve as an example of the API for Submodels in general.
# See packages like [HierarchicalGaussianFiltering.jl](https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl) or [ActiveInference.jl](https://github.com/ComputationalPsychiatry/ActiveInference.jl) for more complex examples of Submodels.

# ## Creating the Submodel type
# First we load the ActionModels package
using ActionModels

# Then we create a Rescorla-Wagner submodel type, which should be passed to the `ActionModel` constructor.
# The Rescorla-Wagner model model has two parameters: the initial value $V_0$ and the learning rate $\alpha$.
# (Note that other parameters, such as the action noise $\beta$, are not part of the Rescorla-Wagner perceptual model, but rather part of the decision model that one can use with it, and therefore are not included in the submodel).
Base.@kwdef struct MyRescorlaWagner <: ActionModels.AbstractSubmodel
    initial_value::Float64 = 0.0
    learning_rate::Float64 = 0.1
end;

# We additionally need to define methods for the two following functions, which are used internally in ActionModels to ensure type stability.
# The two functions need to return the types of the parameters and the states of the submodel, respectively, as `NamedTuple` objects.
function ActionModels.get_parameter_types(submodel::MyRescorlaWagner)
    return (initial_value = Float64, learning_rate = Float64)
end
function ActionModels.get_state_types(submodel::MyRescorlaWagner)
    return (; expected_value = Float64)
end

# ## Creating the model attributes object
# Next, we define an attributes struct that will hold the state and parameters of the Rescorla-Wagner model.
# This struct will be what is passed to the action model function, and which will have custom methods defined for it depending on the model type.
# The attributes object, or whichvever subpart of it that will contain the states and parameters, needs to be mutable, as it will be updated during the model fitting process.
# Additionally, in order to function with Turing, the attributes struct needs to have type parameters that are subtypes of Real for continuous (Float64) and discrete (Int64) parameters and states.
# This is because some AD backends like ForwardDiff and ReverseDiff need to switch between different subtypes of Real to calculate gradients.
# Notably, users may in some situations want to use the same type for the attributes object as for the submodel object above. This is possible, since `AbstractSubmodelAttributes` is a subtype of `AbstractSubmodel`.

Base.@kwdef mutable struct RescorlaWagnerAttributes{T<:Real} <:
                           ActionModels.AbstractSubmodelAttributes
    expected_value::T #the expected value of the observation, which is a state
    initial_value::T  #the parameter which sets the initial value on reset
    learning_rate::T  #the learning rate parameter
end

# We then need to define a method for the `initialize_attributes` function.
# This function is called internally by ActionModels to initialize the attributes of the submodel.
# It needs to take the submodel as the first argument, and then two optional type parameters for continuous and discrete attributes.
# TF stands for the Float64 type, and TI stands for the Int64 type.
# The appropriate types (which here is only the Float64 type) should be used to set the type of the container for the parameters and states.
# In this case, this is the whole attributes struct, but it might also be nested containers.
# This allows ForwardDiff and ReverseDiff to work with the attributes struct, as they need to be able to differentiate through the model.
# If parameters or states are multivariate, the attributes should be initialized with Array{TF} or Array{TI} types instead.
# Additionally, the submodel struct can contain the default parameters, so that they can be passed during the initialization function.
function ActionModels.initialize_attributes(
    submodel::MyRescorlaWagner,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {TF,TI}

    RescorlaWagnerAttributes{TF}(
        expected_value = submodel.initial_value, #the starting expectation is set by the initial value parameter
        initial_value = submodel.initial_value,  #the parameter which sets the initial value on reset
        learning_rate = submodel.learning_rate,  #the learning rate parameter
    )
end

# ## Parameter and state API
# Finally, we need to define methods for getting and setting states and parameters of the submodel, as well as a `reset!` function.
# These allow the outer layer of ActionModels to interact with the submodel in a type-stable way.

# First, we create a method for the `reset!` function, which is called to reset the submodel to its initial state.

function ActionModels.reset!(attributes::RescorlaWagnerAttributes)
    #Set the expected value by the initial value parameter
    attributes.expected_value = attributes.initial_value
    return nothing
end

# Then we create functions for getting the parameters and states from the attributes as `NamedTuple` objects.
function ActionModels.get_parameters(attributes::RescorlaWagnerAttributes)
    return (;
        learning_rate = attributes.learning_rate,
        initial_value = attributes.initial_value,
    )
end
function ActionModels.get_states(attributes::RescorlaWagnerAttributes)
    return (; expected_value = attributes.expected_value)
end

# We also create functions for getting specific parameters and states. These functions should take the attributes as the first argument, and the parameter or state name as the second.
# If the parameter or state name is incorrect, these methods should return an AttributeError, which will be handled by the higher level function.
function ActionModels.get_parameters(
    attributes::RescorlaWagnerAttributes,
    parameter_name::Symbol,
)
    if parameter_name in [:learning_rate, :initial_value]
        return getfield(attributes, parameter_name)
    else
        return AttributeError()
    end
end
function ActionModels.get_states(attributes::RescorlaWagnerAttributes, state_name::Symbol)
    if state_name in [:expected_value]
        return getfield(attributes, state_name)
    else
        return AttributeError()
    end
end

# We also define methods for setting the parameters and states in the attributes.
# These methods should take the attributes as the first argument, the parameter or state name as the second, and the value to set as the third.
# Note that the value passed should be a subtype of `Real`, or a subtype of `AbstractArray{R} where {R<:Real}` if the parameter or state is multivariate.
# If the parameter or state name is incorrect, these methods should return an `AttributeError`, which will be handled by the higher level function.
# If they are found, they should set the value and return `true`.
function ActionModels.set_parameters!(
    attributes::RescorlaWagnerAttributes,
    parameter_name::Symbol,
    parameter_value::T,
) where {T<:Real}
    if parameter_name in [:learning_rate, :initial_value]
        setfield!(attributes, parameter_name, parameter_value)
        return true
    else
        return AttributeError()
    end
end
function ActionModels.set_states!(
    attributes::RescorlaWagnerAttributes,
    state_name::Symbol,
    state_value::T,
) where {T<:Real}
    if state_name in [:expected_value]
        setfield!(attributes, state_name, state_value)
        return true
    else
        return AttributeError()
    end
end

# The `MyRescorlaWagner` submodel and its accompanying `RescorlaWagnerAttributes` attributes struct can now interface with the rest of ActionModels.

# ## Adding custom functionality for the submodel attributes
# Now we have a functioning submodel which can integrate with the rest of ActionModels.
# From here, we can add functionality that can be used with the model attributes inside the action model. 
# In the case of the Rescorla-Wagner model, we add a function which updates the expected value based on an observation, using the Rescorla-Wagner rule.
# For using a submodel to be worth the bother of creating it, there should usually be some more functionality than this.

function update!(attributes::RescorlaWagnerAttributes, observation::Float64)
    #Equation: Vₜ = Vₜ₋₁ + α * (observation - Vₜ₋₁)
    attributes.expected_value +=
        attributes.learning_rate * (observation - attributes.expected_value)
end;

# ## Defining an action model
# We can now define an action model function as usual, which uses the Rescorla-Wagner submodel.
# The submodel attributes can be found in the `submodel` field of the `ModelAttributes` object.
# We here use the `update!()` function we defined above to update the expected value based on the observation.
# In complex models, a function using a submodel like this can greatly simplify the interface when creating action models.

model_function = function rescorla_wagner_gaussian_report(
    attributes::ModelAttributes,
    observation::Float64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Extract Rescorla-Wagner attributes
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the observation
    update!(rescorla_wagner, observation)

    #Extract the expected value from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Return the action distribution, which is a Gaussian with the expected value Vₜ as mean and the action noise β as standard deviation
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end;

# Now we can create the `ActionModel` object. We pass the Rescorla-Wagner submodel as the `submodel` argument, and the model function as the first argument.
# We still need to define the action noise parameter in this call, since it is not part of the Rescorla-Wagner submodel.
# We also still need to define the observation and action, as usual.

action_model = ActionModel(
    model_function,
    submodel = MyRescorlaWagner(learning_rate = 0.1, initial_value = 0.0),
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation()),
    actions = (; report = Action(Normal)),
)

# ## Using the model
# We can now use the action model as usual.
# This includes simulation:

#Initialise an agent
agent = init_agent(action_model, save_history = true)

#Set parameters
set_parameters!(agent, (; action_noise = 0.2, learning_rate = 0.5))

#Define observations
observations = collect(0:0.1:2) .+ randn(21) * 0.1

#Simulate behaviour
simulated_actions = simulate!(agent, observations)

#Plot the expected value trajectory
using StatsPlots
plot(agent, :expected_value, label = "expected value", color = :green)
plot!(observations, label = "observation", color = :red, linetype = :scatter)
plot!(simulated_actions, label = "reported expectation", color = :blue, linetype = :scatter)
ylabel!("Value")
title!("Rescorla-Wagner expectation trajectory")

# And it includes fitting the model to data:

#Create data
using DataFrames
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
    id = vcat(
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
    ),
    treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
)

#Create population model
population_model = [
    Regression(@formula(learning_rate ~ treatment + (1 | id)), logistic),
    Regression(@formula(action_noise ~ treatment + (1 | id)), exp),
]

#Create full model
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)

#Fit model
chns = sample_posterior!(model)

#Plot posterior
plot(chns[[Symbol("learning_rate.β[1]"), Symbol("learning_rate.β[2]"), Symbol("action_noise.β[1]"), Symbol("action_noise.β[2]")]])
