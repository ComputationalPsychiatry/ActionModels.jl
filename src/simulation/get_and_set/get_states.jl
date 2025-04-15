"""
    get_states(agent::Agent, target_state::Union{Symbol,Tuple})

Get a single state from an agent. Returns a single value.

    get_states(agent::Agent, target_state::Vector)

Get a set of state values from an agent. Returns a dictionary of state names and their values.

    get_states(agent::Agent)

Get all states from an agent. Returns a dictionary of state names and their values.
"""
function get_states end


### Functions for getting a single state
function get_states(agent::Agent, target_state::Union{Symbol,Tuple})
    #If the state is in the agent's states
    if target_state in keys(agent.states)
        #Extract it from the agent
        state = agent.states[target_state]
        #Otherwise
    else
        #Look in the submodel
        state = get_states(agent.submodel, target_state)
    end

    return state
end

function get_states(submodel::Nothing, target_state::Union{Symbol,Tuple})
    throw(
        ArgumentError(
            "The specified state $state_name does not exist in the agent or in the submodelure",
        ),
    )

end


### Function for getting multiple states ###
function get_states(agent::Agent, target_states::Vector)

    #Initialize tuple for populating with states
    states = Dict()

    #Go through each state name
    for state_name in target_states
        #Add its value to the list
        states[state_name] = get_states(agent, state_name)
    end

    return states
end


### Function for getting all of an agent's states
function get_states(agent::Agent)

    #Get all state names for the agent
    target_states = collect(keys(agent.states))

    #Get the agent's states 
    agent_states = get_states(agent, target_states)

    #Get states from the submodel
    submodel_states = get_states(agent.submodel)

    #Merge into one list
    states = merge(submodel_states, agent_states)

    return states
end

function get_states(submodel::Nothing)
    return Dict()
end
