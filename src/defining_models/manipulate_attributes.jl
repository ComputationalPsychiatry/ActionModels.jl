#######################################################
####### FUNCTIONS FOR MANIPULATING ATTRIBUTES #########
#######################################################
## Function for resetting states to their initial values ##
function reset!(model_attributes::ModelAttributes)
    #Go through each state
    for (state, initial_state) in
        zip(model_attributes.states, model_attributes.initial_states)
        #If the state is a parameter dependent state, set it to the value of the parameter
        if initial_state isa Variable
            state.value = initial_state.value
        else
            state.value = initial_state
        end
    end

    #Reset the submodel
    reset!(model_attributes.submodel)
end

## Function for setting a single model attribute ##
#For parameters
function set_parameters!(
    model_attributes::ModelAttributes,
    parameter_name::Symbol,
    parameter_value::Union{R,AbstractArray{R}},
) where {R<:Real}

    #Check if the parameter exists in the model attributes
    if haskey(model_attributes.parameters, parameter_name)
        model_attributes.parameters[parameter_name].value = parameter_value
    else
        #Set it in the submodel
        out = set_parameters!(model_attributes.submodel, parameter_name, parameter_value)
        if out isa AttributeError
            error(
                "Parameter $parameter_name not found in model attributes or in the submodel.",
            )
        end
    end
end
#For states
function set_states!(
    model_attributes::ModelAttributes,
    state_name::Symbol,
    state_value::S,
) where {S}

    #Check if the state exists in the model attributes
    if haskey(model_attributes.states, state_name)
        model_attributes.states[state_name].value = state_value
    else
        out = set_states!(model_attributes.submodel, state_name, state_value)
        if out isa AttributeError
            error("State $state_name not found in model attributes or in the submodel.")
        end
    end
end
#For actions
function set_actions!(
    model_attributes::ModelAttributes,
    action_name::Symbol,
    action::A,
) where {A<:Any}
    model_attributes.actions[action_name].value = action
    return nothing
end

## Functions for setting multiple attributes ##
#Parameters
function set_parameters!(
    model_attributes::ModelAttributes,
    parameter_names::Tuple{Vararg{Symbol}},
    parameters::Tuple{Vararg{Union{R,AbstractArray{R}}}} where {R<:Real},
)
    for (parameter_name, parameter_value) in zip(parameter_names, parameters)
        set_parameters!(model_attributes, parameter_name, parameter_value)
    end
end
#States
function set_states!(
    model_attributes::ModelAttributes,
    state_names::Tuple{Vararg{Symbol}},
    states::Tuple{Vararg{Any}},
)
    for (state_name, state_value) in zip(state_names, states)
        set_states!(model_attributes, state_name, state_value)
    end
end
#For multiple actions
function set_actions!(
    model_attributes::ModelAttributes,
    action_names::Tuple{Vararg{Symbol}},
    actions::Tuple{Vararg{Real}},
)
    for (action_name, action) in zip(action_names, actions)
        model_attributes.actions[action_name].value = action
    end
end



#####################################################
####### FUNCTIONS FOR EXTRACTING ATTRIBUTES #########
#####################################################
## Function for extracting all model attributes ##
function get_parameters(model_attributes::ModelAttributes)
    return merge(
        map(parameter -> parameter.value, model_attributes.parameters),
        get_parameters(model_attributes.submodel),
    )
end
function get_states(model_attributes::ModelAttributes)
    return merge(
        map(state -> state.value, model_attributes.states),
        get_states(model_attributes.submodel),
    )
end
function get_actions(model_attributes::ModelAttributes)
    return map(action -> action.value, model_attributes.actions)
end

## Functions for extracting multiple model attributes ##
function get_parameters(
    model_attributes::ModelAttributes,
    parameter_names::Tuple{Vararg{Symbol}},
)
    return NamedTuple(
        parameter_name => get_parameters(model_attributes, parameter_name) for
        parameter_name in parameter_names
    )
end
function get_states(model_attributes::ModelAttributes, state_names::Tuple{Vararg{Symbol}})
    return NamedTuple(
        state_name => get_states(model_attributes, state_name) for state_name in state_names
    )
end
function get_actions(model_attributes::ModelAttributes, action_names::Tuple{Vararg{Symbol}})
    return NamedTuple(
        action_name => get_actions(model_attributes, action_name) for
        action_name in action_names
    )
end

## Functions for extracting a single model attribute ##
function get_parameters(model_attributes::ModelAttributes, parameter_name::Symbol)
    #Check if the parameter exists in the model attributes
    if haskey(model_attributes.parameters, parameter_name)
        return model_attributes.parameters[parameter_name].value
    else
        #Set it in the submodel
        parameter = get_parameters(model_attributes.submodel, parameter_name)
        if parameter isa AttributeError
            error(
                "Parameter $parameter_name not found in model attributes or in the submodel.",
            )
        end
        return parameter
    end
end
function get_states(model_attributes::ModelAttributes, state_name::Symbol)
    #Check if the state exists in the model attributes
    if haskey(model_attributes.states, state_name)
        return model_attributes.states[state_name].value
    else
        #Set it in the submodel
        state = get_states(model_attributes.submodel, state_name)
        if state isa AttributeError
            error("State $state_name not found in model attributes or in the submodel.")
        end
        return state
    end
end
function get_actions(model_attributes::ModelAttributes, action_name::Symbol)
    if haskey(model_attributes.actions, action_name)
        return model_attributes.actions[action_name].value
    else
        error("Action $action_name not found in model attributes.")
    end
end
