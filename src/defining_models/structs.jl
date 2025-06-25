############################################
### TYPE FOR INITIALIZING AN ACTIONMODEL ###
############################################
## Supertype for model attributes ##
abstract type AbstractAttribute end

## For creating parameters ##
abstract type AbstractParameter <: AbstractAttribute end

"""
Parameter(value; discrete=false)

Type constructor for defining parameters in action models. Can be continuous or discrete.

# Arguments
- `value`: The default value of the parameter. Can be a single value or an array.
- `discrete`: If true, parameter is treated as discrete (Int), otherwise as continuous (Float).

# Examples
```jldoctest
julia> Parameter(0.1)
Parameter{Float64}(0.1, Float64)

julia> Parameter(1, discrete=true)
Parameter{Int64}(1, Int64)

julia> Parameter([0.0, 0.0])
Parameter{Array{Float64}}([0.0, 0.0], Array{Float64})
```
"""
struct Parameter{T<:Union{Real,Array{<:Real}}} <: AbstractParameter
    value::T
    type::Type{T}

    function Parameter(
        value::T;
        discrete::Bool = false,
    ) where {T<:Union{Real,Array{<:Real}}}
        if discrete
            if value isa Array
                new{Array{Int64}}(value, Array{Int64})
            else
                new{Int64}(value, Int64)
            end
        else
            if value isa Array
                new{Array{Float64}}(value, Array{Float64})
            else
                new{Float64}(value, Float64)
            end
        end
    end
end

"""
InitialStateParameter(initial_value, state_name; discrete=false)

Type for defining initial state parameters in action models. Initial state parameters define the initial value of a state in the model. Can be continuous or discrete.

# Arguments
- `value`: The default value for the initial state parameter. Can be a single value or an array.
- `state_name`: The symbol name of the state this parameter controls.
- `discrete`: If true, value is treated as discrete (Int), otherwise as continuous (Float).

# Examples
```jldoctest
julia> InitialStateParameter(0.0, :expected_value)
InitialStateParameter{Float64}(:expected_value, 0.0, Float64)

julia> InitialStateParameter(1, :counter, discrete=true)
InitialStateParameter{Int64}(:counter, 1, Int64)

julia> InitialStateParameter([0.0, 0.0], :weights)
InitialStateParameter{Array{Float64}}(:weights, [0.0, 0.0], Array{Float64})
```
"""
struct InitialStateParameter{T<:Union{Real,Array{<:Real}}} <: AbstractParameter
    state::Symbol
    value::T
    type::Type{T}
    function InitialStateParameter(
        value::T,
        state_name::Symbol;
        discrete::Bool = false,
    ) where {T<:Union{Real,Array{<:Real}}}
        if discrete
            if value isa Array
                new{Array{Int64}}(state_name, value, Array{Int64})
            else
                new{Int64}(state_name, value, Int64)
            end
        else
            if value isa Array
                new{Array{Float64}}(state_name, value, Array{Float64})
            else
                new{Float64}(state_name, value, Float64)
            end
        end
    end
end

## For creating states ##
abstract type AbstractState <: AbstractAttribute end

"""
State(initial_value; discrete=nothing)
State(initial_value, ::Type{T})
State(; discrete=false)
State(::Type{T})

Construct a model state variable, which can be continuous, discrete, or a custom type.

# Arguments
- `initial_value`: The initial value for the state (can be Real, Array, or custom type). Set to `missing` for no initial value.
- `discrete`: If true, state is treated as discrete (Int), otherwise as continuous (Float). Only valid for Real or Array{<:Real} types.
- `T`: Type of the state (for non-Real types).

# Examples
```jldoctest
julia> State(0.0)
State{Float64}(0.0, Float64)

julia> State(1, discrete=true)
State{Int64}(1, Int64)

julia> State([0.0, 0.0])
State{Array{Float64}}([0.0, 0.0], Array{Float64})

julia> State(discrete=true)
State{Int64}(missing, Int64)

julia> State(String)
State{String}(missing, String)
```
"""
struct State{T} <: AbstractState
    initial_value::Union{Missing,T}
    type::Type{T}

    function State(initial_value; discrete::Union{Nothing,Bool} = nothing)

        #If a non-real value has been specified
        if !(initial_value isa Real) && !(initial_value isa Array{R} where {R<:Real})

            if !isnothing(discrete)
                throw(
                    ArgumentError(
                        "The discrete keyword is only defined for Real or Array{<:Real} types initial values.",
                    ),
                )
            end

            return new{typeof(initial_value)}(initial_value, typeof(initial_value))
        else
            #If discrete is not specified, set it to false
            if isnothing(discrete)
                discrete = false
            end
        end

        if discrete
            if initial_value isa Array
                return new{Array{Int64}}(initial_value, Array{Int64})
            else
                return new{Int64}(initial_value, Int64)
            end
        else
            if initial_value isa Array
                return new{Array{Float64}}(initial_value, Array{Float64})
            else
                return new{Float64}(initial_value, Float64)
            end
        end
    end

    function State(initial_value, ::Type{T}) where {T}
        new{T}(initial_value, T)
    end

    function State(; discrete::Bool = false)
        if discrete
            return new{Int64}(missing, Int64)
        else
            return new{Float64}(missing, Float64)
        end
    end

    function State(::Type{T}) where {T}
        new{T}(missing, T)
    end

end

## For creating observations ##
abstract type AbstractObservation <: AbstractAttribute end

"""
Observation(; discrete=false)
Observation([T])

Construct an observation input to the model. Can be continuous (Float64), discrete (Int64), or a custom type.

# Arguments
- `discrete`: If true, observation is treated as discrete (Int64), otherwise as continuous (Float64).
- `T`: Type of the observation (e.g., Float64, Int64, Vector{Float64}). Used for setting specific types.

# Examples
```jldoctest
julia> Observation()
Observation{Float64}(Float64)

julia> Observation(discrete=true)
Observation{Int64}(Int64)

julia> Observation(Vector{Float64})
Observation{Vector{Float64}}(Vector{Float64})

julia> Observation(String)
Observation{String}(String)
```
"""
struct Observation{T} <: AbstractObservation
    type::Type{T}

    function Observation(::Type{T}) where {T}
        return new{T}(T)
    end

    function Observation(; discrete::Bool = false)
        if discrete
            return new{Int64}(Int64)
        else
            return new{Float64}(Float64)
        end
    end
end

## For creating actions ##
abstract type AbstractAction <: AbstractAttribute end

"""
Action(distribution_type, action_type)

Construct an action output for the model, specifying the distribution type (e.g., Normal, Bernoulli).

# Arguments
- `T`: Type of the action (optional, inferred from distribution_type if not given).
- `distribution_type`: The distribution type for the action (e.g., Normal, Bernoulli, MvNormal). Can also be an abstract type like `Distribution{Multivariate, Continuous}` to allow multiple types of distributions.

# Examples
```jldoctest
julia> Action(Normal)
Action{Float64, Normal}(Float64, Normal)

julia> Action(Bernoulli)
Action{Int64, Bernoulli}(Int64, Bernoulli)

julia> Action(MvNormal)
Action{Array{Float64}, MvNormal}(Array{Float64}, MvNormal)

julia> Action(Distribution{Multivariate, Discrete})
Action{Array{Int64}, Distribution{Multivariate, Discrete}}(Array{Int64}, Distribution{Multivariate, Discrete})
```
"""
struct Action{T,TD} <: AbstractAction
    type::Type{T}
    distribution_type::Type{TD}

    function Action(
        distribution_type::Type{TD},
        action_type::Type{T},
    ) where {T<:Union{Real,Array{<:Real}},TD<:Distribution}

        if !(get_action_type(TD) <: T)
            @warn "Action type $T is not a supertype of $(get_action_type(TD)), which is the type that matches the chosen distribution type $TD. Check that everything is in order"
        end
        new{T,TD}(action_type, distribution_type)
    end

    function Action(distribution_type::Type{TD}) where {TD<:Distribution}

        action_type = get_action_type(TD)

        T = action_type

        new{T,TD}(action_type, distribution_type)
    end
end

## Function to get the action type based on the distribution type ##
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Univariate,Continuous}}
    return Float64
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Multivariate,Continuous}}
    return Array{Float64}
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Univariate,Discrete}}
    return Int64
end
function get_action_type(
    action_dist_type::Type{T},
) where {T<:Distribution{Multivariate,Discrete}}
    return Array{Int64}
end


## Supertype for submodels ##
abstract type AbstractSubmodel end

"""
NoSubModel()

Default submodel type used when no submodel is specified in an ActionModel.

# Examples
```jldoctest
julia> ActionModels.NoSubModel()
ActionModels.NoSubModel()
```
"""
struct NoSubModel <: AbstractSubmodel end

## ActionModel struct ##
abstract type AbstractActionModel end

"""
ActionModel(action_model; parameters, states, observations, actions, submodel, verbose)

Main container for a user-defined or premade action model.

# Arguments
- `action_model`: The function implementing the model's update and action logic.
- `parameters`: NamedTuple of model parameters or a single Parameter.
- `states`: NamedTuple of model states or a single State (optional).
- `observations`: NamedTuple of model observations or a single Observation (optional).
- `actions`: NamedTuple of model actions or a single Action.
- `submodel`: Optional submodel for hierarchical or modular models (default: NoSubModel()).
- `verbose`: Print warnings for singletons (default: true).

# Examples
```jldoctest
julia> model_fn = (attributes, obs) -> Normal(0, 1);

julia> ActionModel(model_fn, parameters=(learning_rate=Parameter(0.1),), states=(expected_value=State(0.0),), observations=(observation=Observation(),), actions=(report=Action(Normal),))
-- ActionModel --
Action model function: #1
Number of parameters: 1
Number of states: 1
Number of observations: 1
Number of actions: 1

julia> ActionModel(model_fn, parameters=Parameter(0.1), actions = Action(Normal), verbose = false)
-- ActionModel --
Action model function: #1
Number of parameters: 1
Number of states: 0
Number of observations: 0
Number of actions: 1
```
"""
struct ActionModel{T<:AbstractSubmodel} <: AbstractActionModel
    action_model::Function
    parameters::NamedTuple{
        parameter_names,
        <:Tuple{Vararg{AbstractParameter}},
    } where {parameter_names}
    states::NamedTuple{state_names,<:Tuple{Vararg{AbstractState}}} where {state_names}
    observations::NamedTuple{
        observation_names,
        <:Tuple{Vararg{AbstractObservation}},
    } where {observation_names}
    actions::NamedTuple{action_names,<:Tuple{Vararg{AbstractAction}}} where {action_names}
    submodel::T

    function ActionModel(
        action_model::Function;
        parameters::Union{
            AbstractParameter,
            NamedTuple{parameter_names,<:Tuple{Vararg{AbstractParameter}}},
        } where {parameter_names},
        states::Union{
            AbstractState,
            NamedTuple{state_names,<:Tuple{Vararg{AbstractState}}},
        } where {state_names} = (;),
        observations::Union{
            AbstractObservation,
            NamedTuple{observation_names,<:Tuple{Vararg{AbstractObservation}}},
        } where {observation_names} = (;),
        actions::Union{
            AbstractAction,
            NamedTuple{action_names,<:Tuple{Vararg{AbstractAction}}},
        } where {action_names},
        submodel::T = NoSubModel(),
        verbose::Bool = true,
    ) where {T<:AbstractSubmodel}
        #Make single structs into NamedTuples
        if parameters isa AbstractParameter
            parameters = (; parameter = parameters)
            if verbose
                @warn "A single parameter was passed, and is given the name :parameter. Use (; parameter_name = parameter) to specify the name of the parameter."
            end
        end
        if states isa AbstractState
            states = (; state = states)
            if verbose
                @warn "A single state was passed, and is given the name :state. Use (; state_name = state) to specify the name of the state."
            end
        end
        if actions isa AbstractAction
            actions = (; action = actions)
            if verbose
                @warn "A single action was passed, and is given the name :action. Use (; action_name = action) to specify the name of the action."
            end
        end
        if observations isa AbstractObservation
            observations = (; observation = observations)
            if verbose
                @warn "A single observation was passed, and is given the name :observation. Use (; observation_name = observation) to specify the name of the observation."
            end
        end

        #Check initial state parameters
        for (parameter_name, parameter) in pairs(parameters)
            if parameter isa InitialStateParameter
                #Check that the parameter sets a state that exists
                state_name = parameter.state
                if !(state_name in keys(states))
                    throw(
                        ArgumentError(
                            "The initial state parameter $parameter_name sets the state $state_name, but this state has not been specified by the user.",
                        ),
                    )
                end
                #Check that the type of the parameter matches the type of the state
                state = states[state_name]
                if !(parameter.type <: state.type)
                    throw(
                        ArgumentError(
                            "The initial state parameter $parameter_name has type $(parameter.type), but sets the state $state_name which has type $(state.type).",
                        ),
                    )
                end
            end
        end

        return new{T}(action_model, parameters, states, observations, actions, submodel)
    end
end



#########################################
### TYPES FOR USE IN THE ACTION MODEL ###
#########################################
## Abstract type for submodel attributes ##
# It is a subtype of AbstractSubmodel, so that attributes can be as Submodels too
abstract type AbstractSubmodelAttributes <: AbstractSubmodel end

struct NoSubModelAttributes <: AbstractSubmodelAttributes end

"""
ModelAttributes(parameters, states, actions, initial_states, submodel)

Internal container for all model variables and submodel attributes used in an action model instance.

# Arguments
- `parameters`: NamedTuple of parameter variables 
- `states`: NamedTuple of state variables 
- `actions`: NamedTuple of action variables 
- `initial_states`: NamedTuple of initial state values
- `submodel`: Submodel attributes (or `NoSubModelAttributes` if not used)

# Examples
```jldoctest
julia> ModelAttributes((learning_rate=ActionModels.Variable(0.1),), (expected_value=ActionModels.Variable(0.0),), (report=ActionModels.Variable(missing),), (expected_value=ActionModels.Variable(0.0),), ActionModels.NoSubModelAttributes())
ModelAttributes{@NamedTuple{learning_rate::ActionModels.Variable{Float64}}, @NamedTuple{expected_value::ActionModels.Variable{Float64}}, @NamedTuple{report::ActionModels.Variable{Missing}}, @NamedTuple{expected_value::ActionModels.Variable{Float64}}, ActionModels.NoSubModelAttributes}((learning_rate = ActionModels.Variable{Float64}(0.1),), (expected_value = ActionModels.Variable{Float64}(0.0),), (report = ActionModels.Variable{Missing}(missing),), (expected_value = ActionModels.Variable{Float64}(0.0),), ActionModels.NoSubModelAttributes())
```
"""
struct ModelAttributes{
    TP<:NamedTuple,
    TS<:NamedTuple,
    TA<:NamedTuple,
    IS<:NamedTuple,
    TM<:AbstractSubmodelAttributes,
}
    parameters::TP
    states::TS
    actions::TA
    initial_states::IS
    submodel::TM
end

"""
Variable(value)

A mutable container for a value of type `T`. Used for model parameters, states, and actions.

# Arguments
- `value`: The value to store (any type)

# Examples
```jldoctest
julia> v = ActionModels.Variable(0.5)
ActionModels.Variable{Float64}(0.5)

julia> v.value = 1.0
1.0

julia> v
ActionModels.Variable{Float64}(1.0)
```
"""
mutable struct Variable{T}
    value::T
end

struct ParameterDependentState
    parameter_name::Symbol
end

"""
RejectParameters(errortext)

Custom error type for rejecting parameter samples during inference.

# Arguments
- `errortext`: Explanation for the rejection

# Examples
```jldoctest
julia> throw(RejectParameters("Parameter out of bounds"))
ERROR: RejectParameters("Parameter out of bounds")
```
"""
struct RejectParameters <: Exception
    errortext::Any
end

"""
AttributeError()

Custom error type for missing or invalid attribute access. Used for error handling in custom submodels.

# Examples
```jldoctest
julia> throw(AttributeError())
ERROR: AttributeError()
```
"""
struct AttributeError <: Exception end



###################
### OTHER TYPES ###
###################
## Supertype for premade models ##
abstract type AbstractPremadeModel end
