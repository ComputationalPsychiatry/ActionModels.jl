#############################################
####### INITIALIZING MODEL ATTRIBUTES #######
#############################################
## For initializing the model attributes ##
function initialize_attributes(
    action_model::ActionModel,
    initial_states::NamedTuple{initial_state_keys,<:Tuple{Vararg{Any}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {initial_state_keys,TF,TI}

    parameters = initialize_variables(action_model.parameters, TF, TI)

    states = initialize_variables(action_model.states, TF, TI)

    actions = initialize_variables(action_model.actions, TF, TI)

    initial_states = map(
        state ->
            state isa ParameterDependentState ? parameters[state.parameter_name] : state,
        initial_states,
    )

    submodel_attributes = initialize_attributes(action_model.submodel, TF, TI)

    return ModelAttributes(parameters, states, actions, initial_states, submodel_attributes)
end


######################################################################
####### FUNCTION FOR LOADING TYPE FROM THE TURING MODEL HEADER #######
######################################################################
## Intializing variables with correct types for an attribute set ##
function initialize_variables(
    parameters::NamedTuple{names,<:Tuple{Vararg{AbstractParameter}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return map(
        parameter -> Variable{load_type(parameter.type, TF, TI)}(parameter.value),
        parameters,
    )
end
function initialize_variables(
    states::NamedTuple{names,<:Tuple{Vararg{AbstractState}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return map(
        state ->
            Variable{Union{Missing,load_type(state.type, TF, TI)}}(state.initial_value),
        states,
    ) #TODO: only allow missing for states with missing initial values?
end
function initialize_variables(
    actions::NamedTuple{names,<:Tuple{Vararg{AbstractAction}}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return map(
        action -> Variable{Union{Missing,load_type(action.type, TF, TI)}}(missing),
        actions,
    )
end
#And if it is an empty collection
function initialize_variables(
    states::NamedTuple{names,<:Tuple{}},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {names,TF,TI}
    return (;)
end

## Load the correct type for a single attribute ##
#For returning an array type
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {ST,T<:Array{ST},TF,TI}
    NT = load_type(ST, TF, TI)
    return Array{NT}
end
#For returning a single type
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {T<:AbstractFloat,TF,TI}
    return TF
end
function load_type(
    ::Type{T},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {T<:Integer,TF,TI}
    return TI
end
function load_type(::Type{T}, ::Type{TF} = Float64, ::Type{TI} = Int64) where {T,TF,TI}
    return T
end



#################################################
####### INTERNAL MANIPULATION FUNCTIONS #########
#################################################
## Internal manipulation functions ##
#Store a single sampled action - used in model fitting and simulation
function store_action!(model_attributes::ModelAttributes, sampled_action::A) where {A<:Real}
    first(model_attributes.actions).value = sampled_action
end
#Store a set of actions from one timestep - used in model fitting and simulation
function store_actions!(
    model_attributes::ModelAttributes,
    sampled_actions::Tuple{Vararg{Real}},
)
    for (action, sampled_action) in zip(model_attributes.actions, sampled_actions)
        action.value = sampled_action
    end
end

#Update a state with a new value - used within action model definition
function update_state!(
    model_attributes::ModelAttributes,
    state_name::Symbol,
    state_value::S,
) where {S}
    model_attributes.states[state_name].value = state_value
end

#Extracting parameters, states and actions from the model attributes - used within action model definition
function load_parameters(model_attributes::ModelAttributes)
    return map(parameter -> parameter.value, model_attributes.parameters)
end
function load_states(model_attributes::ModelAttributes)
    return map(state -> state.value, model_attributes.states)
end
function load_actions(model_attributes::ModelAttributes)
    return map(action -> action.value, model_attributes.actions)
end