"""
    single_input!(agent::Agent, input::Any)

Give a single input to an Agent and let it evolve. Returns the agent's action.
"""
function single_input!(agent::Agent, input::Any)

    #Run the action model to get the action distribution
    action_distribution = agent.action_model(agent, input)

    #If a single action distribution is returned
    if length(action_distribution) == 1

        #Sample an action from the distribution
        agent.states[:action] = rand(action_distribution)

        #If multiple action distributions are returned
    else
        #Initialize vector for storing actions
        actions = []

        #For each action distribution
        for distribution in action_distribution
            #Sample an action and add it to the vector
            push!(actions, rand(distribution))
        end

        #And store it 
        agent.states[:action] = actions
    end

    #Record the action
    push!(agent.history[:action], agent.states[:action])

    #Return the action
    return agent.states[:action]
end


"""
    give_inputs!(agent::Agent, inputs)

Give inputs to an agent. Input can be a single value, a vector of values, or an array of values. Returns the agent's action trajectory, without the initial state.
"""
function give_inputs! end

function give_inputs!(agent::Agent, input::Real)

    #Input the single input
    single_input!(agent, input)

    #Get the history of actions, without the initial state
    actions = agent.history[:action][2:end]

    #Make into array, and return it
    return stack(actions, dims = 1)
end

function give_inputs!(
    agent::Agent,
    inputs::Union{VR,VVR},
) where {R<:Real,VR<:Vector{R},VVR<:Vector{VR}}

    #Each value in the vector is a single input
    for input in inputs

        #Input that row
        single_input!(agent, input)

    end

    #Get the history of actions, without the initial state
    actions = agent.history[:action][2:end]

    #Make into array, and return it
    return stack(actions, dims = 1)
end

function give_inputs!(agent::Agent, inputs::Matrix)

    #Each row in the array is a single input
    for input in Tuple.(eachrow(inputs))

        #Input that row
        single_input!(agent, input)

    end

    #Get the history of actions, without the initial state
    actions = agent.history[:action][2:end]

    #Make into array, and return it
    return stack(actions, dims = 1)
end
