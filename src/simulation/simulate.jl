###################################
### MAKING A SINGLE OBSERVATION ###
###################################
"""
    observe!(agent::Agent, observation)

Advance the agent by one timestep using the given observation, updating its state and returning the sampled action.

# Arguments
- `agent`: The `Agent` to update.
- `observation`: The observation(s) for this timestep (tuple or value).

# Returns
- The sampled action(s) for this timestep.

# Examples
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> action = observe!(agent, 0.5);

julia> action isa Real
true
```
"""
function observe!(agent::Agent, observation::T) where {T<:Any}

    #Run the action model to get the action distribution
    action_distribution = agent.action_model(agent.model_attributes, observation...)

    #Sample actions
    action = sample_actions!(action_distribution)

    #Store the action
    store_action!(agent.model_attributes, action)

    #Save states in history
    for (state_name, state_value) in pairs(agent.history)
        push!(state_value, get_states(agent, state_name))
    end

    #Count the timestep
    agent.n_timesteps.value += 1

    #Return the action
    return action
end

## Functions for sampling a single action ##
function sample_actions!(action::D) where {D<:Distributions.Distribution}
    #Sample an action from the action distribution
    return rand(action)
end
function sample_actions!(action::Tuple{Vararg{Distribution}})
    return map(sample_actions!, action)
end


#####################################
### SIMULATING MULTIPLE TIMESTEPS ###
#####################################
## With pre-specified observations ##
#With a vector where each element is a single observation or a tuple of observations
"""
    simulate!(agent::Agent, observations::AbstractVector)

Simulate the agent forward for multiple timesteps, using a vector of observations. Returns a vector of actions for each timestep.

# Arguments
- `agent`: The `Agent` to simulate.
- `observations`: Vector of observations (each element is a tuple or value for one timestep).

# Returns
- Vector of actions, one per timestep.

# Examples
```jldoctest; setup = :(using ActionModels; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true))
julia> actions = simulate!(agent, [0.5, 0.7, 0.9]);

julia> actions isa AbstractVector
true

```
"""
function simulate!(agent::Agent, observations::AbstractVector{T}) where {T<:Any}

    #Simulate forward
    actions = [observe!(agent, observation) for observation in observations]

    #Return the actions
    return actions
end
#With a matrix where each row is a single observation
"""
    simulate!(agent::Agent, observations::AbstractMatrix)

Simulate the agent forward for multiple timesteps, using a matrix of observations (each row is a timestep). Returns a vector of actions.

# Arguments
- `agent`: The `Agent` to simulate.
- `observations`: Matrix of observations (each row is a tuple or value for one timestep).

# Returns
- Vector of actions, one per timestep.
"""
function simulate!(agent::Agent, observations::AbstractMatrix{T}) where {T<:Any}

    #Simulate forward
    actions = [observe!(agent, observation) for observation in eachrow(observations)]

    #Return the actions
    return actions
end

#TODO: with an environment

#TODO: with another agent
