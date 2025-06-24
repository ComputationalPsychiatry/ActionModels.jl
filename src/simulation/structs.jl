"""
    Agent{T<:AbstractSubmodel}

Container for a simulated agent, including the action model, model attributes, state history, and number of timesteps.

# Fields
- `action_model`: The function implementing the agent's action model logic.
- `model_attributes`: The `ModelAttributes` instance containing parameters, states, and actions.
- `history`: NamedTuple of vectors storing the history of selected states.
- `n_timesteps`: Variable tracking the number of timesteps simulated.

# Examples
```jldoctest
julia> am = ActionModel(RescorlaWagner());

julia> agent = init_agent(am, save_history=true)
-- ActionModels Agent --
Action model: rescorla_wagner_act_after_update
This agent has received 0 observations
```
"""
struct Agent{T<:AbstractSubmodel}
    action_model::Function
    model_attributes::ModelAttributes
    history::NamedTuple
    n_timesteps::Variable{Int64}
end

"""
    init_agent(action_model::ActionModel; save_history=false)

Initialize an `Agent` for simulation, given an `ActionModel`. Optionally specify which states to save in the agent's history.

# Arguments
- `action_model`: The `ActionModel` to use for the agent.
- `save_history`: If `true`, save all states; if `false`, save none; if a `Symbol` or `Vector{Symbol}`, save only those states (default: `false`).

# Returns
- `Agent`: An initialized agent ready for simulation.

# Examples
```jldoctest
julia> am = ActionModel(RescorlaWagner());

julia> agent = init_agent(am, save_history=true)
-- ActionModels Agent --
Action model: rescorla_wagner_act_after_update
This agent has received 0 observations
```
"""
function init_agent(
    action_model::ActionModel;
    save_history::Union{Bool,Symbol,Vector{Symbol}} = false,
)
    ## Initialize model attributes ##
    #Find initial states
    initial_state_parameter_state_names = NamedTuple(
        parameter.state => ParameterDependentState(parameter_name) for
        (parameter_name, parameter) in pairs(action_model.parameters) if
        parameter isa InitialStateParameter
    )
    initial_states = NamedTuple(
        state_name in keys(initial_state_parameter_state_names) ?
        state_name => initial_state_parameter_state_names[state_name] :
        state_name => state.initial_value for
        (state_name, state) in pairs(action_model.states)
    )

    #Initialize model attributes
    model_attributes = initialize_attributes(action_model, initial_states)

    #Reset it, so that the initial states are set to the initial values
    reset!(model_attributes)

    ## Initialize history ##
    #Extract state types from action model and submodel
    state_types =
        merge(get_state_types(action_model), get_state_types(action_model.submodel))

    #Find states for which to save history
    if save_history isa Bool
        #If save_history is true, save all states
        if save_history
            save_history = [
                collect(keys(model_attributes.states));
                collect(keys(get_state_types(action_model.submodel)))
            ]
        else
            #If save_history is false, don't save any states
            save_history = Symbol[]
        end
    end
    if save_history isa Symbol
        #If save_history is a symbol, save only that state
        save_history = [save_history]
    end

    #Initialize history
    history = NamedTuple(
        state_name => Vector{state_types[state_name]}() for state_name in save_history
    )

    #Add initial states to history
    for (state_name, state) in pairs(history)
        push!(state, get_states(model_attributes, state_name))
    end

    #Get submodel type
    submodel_type = typeof(action_model.submodel)

    ## Create agent ##
    Agent{submodel_type}(action_model.action_model, model_attributes, history, Variable{Int64}(0))
end
