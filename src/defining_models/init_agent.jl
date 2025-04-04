
"""
    init_agent(action_model::Function; substruct::Any = nothing, parameters::Dict = Dict(), states::Union{Dict, Vector} = Dict(),
    settings::Dict = Dict(), parameter_groups::Dict = Dict())
    
Initialize an agent. 

Note that action_model can also be specified as a vector of action models: action_model::Vector{Function}.
In this case the action models will be stored in the agent's settings. In that case use the function 'multiple_actions'

# Arguments
 - 'action_model::Function': a function specifying the agent's action model. Can be any function that takes an agent and a single input as arguments, and returns a probability distribution from which actions are sampled.
 - 'substruct::Any = nothing': struct containing additional parameters and states. This structure also get passed to utility functions. Check advanced usage guide.
 - 'parameters::Dict = Dict()': dictionary containing parameters of the agent. Keys are parameter names (strings, or tuples of strings), values are parameter values.
 - 'states::Union{Dict, Vector} = Dict()': dictionary containing states of the agent. Keys are state names (strings, or tuples of strings), values are initial state values. Can also be a vector of state name strings.
 - 'settings::Dict = Dict()': dictionary containing additional settings for the agent. Keys are setting names, values are setting values.
 - 'parameter_groups::Dict = Dict()': dictionary containing shared parameters. Keys are the the name of the shared parameter, values are the value of the shared parameter followed by a vector of the parameters sharing that value.
# Examples
```julia
## Create agent with a binary Rescorla-Wagner action model ##

## Create action model function
function binary_rescorla_wagner_softmax(agent::Agent, input::Union{Bool,Integer})

    #Read in parameters
    learning_rate = agent.parameters["learning_rate"]
    action_precision = agent.parameters["action_precision"]

    #Read in states
    old_value = agent.states["value"]

    #Sigmoid transform the value
    old_value_probability = 1 / (1 + exp(-old_value))

    #Get new value state
    new_value = old_value + learning_rate * (input - old_value_probability)

    #Pass through softmax to get action probability
    action_probability = 1 / (1 + exp(-action_precision * new_value))

    #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
    action_distribution = Distributions.Bernoulli(action_probability)

    #Update states
    agent.states["value"] = new_value
    agent.states["value_probability"], 1 / (1 + exp(-new_value))
    agent.states["action_probability"], action_probability
    #Add to history
    push!(agent.history["value"], new_value)
    push!(agent.history["value_probability"], 1 / (1 + exp(-new_value)))
    push!(agent.history["action_probability"], action_probability)

    return action_distribution
end

#Define requried parameters
parameters = Dict(
    "learning_rate" => 1,
    "action_precision" => 1,
    ("initial", "value") => 0,
)

#Define required states
states = Dict(
    "value" => missing,
    "value_probability" => missing,
    "action_probability" => missing,
)

#Create agent
agent = init_agent(
    binary_rescorla_wagner_softmax,
    parameters = parameters,
    states = states,
    settings = settings,
)

"""
function init_agent() end


function init_agent(
    action_model::Function;
    substruct::Any = nothing,
    parameters::Dict = Dict(),
    parameter_groups::Union{ParameterGroup,Vector{ParameterGroup}} = Vector{
        ParameterGroup,
    }(),
    states::Union{Dict,Vector} = Dict(),
    settings::Dict = Dict(),
    save_history::Bool = true,
)

    ##Create action model struct
    agent = Agent(
        action_model = action_model,
        substruct = substruct,
        parameters = Dict(),
        initial_state_parameters = Dict(),
        initial_states = Dict(),
        states = Dict(),
        settings = settings,
        save_history = save_history,
    )


    ##Add parameters to either initial state parameters or parameters
    for (param_key, param_value) in parameters
        
        #If the param is an initial state parameter
        if param_value isa InitialState

            #Add the parameter using the state as key
            agent.initial_state_parameters[param_key] = param_value
            agent.initial_states[param_value.state] = param_value

        else
            #For other parameters, add to parameters
            agent.parameters[param_key] = param_value
        end
    end


    ##Add states
    #If states is a dictionary
    if states isa Dict
        #Insert as states
        agent.states = states
        #If states is a vector
    elseif states isa Vector
        #Go through each state
        for state in states
            #And set to missing
            agent.states[state] = missing
        end
    end

    #If an action state was not specified
    if !("action" in keys(agent.states))
        #Add an empty action state
        agent.states["action"] = missing
    end


    #If there is only one parameter group, wrap it in a vector
    if parameter_groups isa ParameterGroup
        parameter_groups = [parameter_groups]
    end

    #Go through each specified shared parameter
    for parameter_group in parameter_groups

        #check if the name of the shared parameter is part of its own derived parameters
        if parameter_group.name in parameter_group.parameters
            throw(
                ArgumentError(
                    "The shared parameter $parameter_group is among the parameters it is defined to set",
                ),
            )
        end

        #Set the parameter group in the agent
        agent.shared_parameters[parameter_group.name] = GroupedParameters(
            value = parameter_group.value,
            grouped_parameters = parameter_group.parameters,
        )

        #Set the parameters 
        set_parameters!(agent, parameter_group, parameter_group.value)

    end

    #Reset the substruct to make sure initial states are correct, after setting the grouped parameters
    reset!(substruct)


    #Initialize states
    for (param_key, initial_state) in agent.initial_state_parameters

        #Extract the state and value
        state_key = initial_state.state
        initial_value = initial_state.value

        #If the state exists
        if state_key in keys(agent.states)
            #Set initial state
            agent.states[state_key] = initial_value

        else
            #Throw error
            throw(
                ArgumentError(
                    "The state $(state_key) has an initial state parameter, but does not exist in the agent.",
                ),
            )
        end
    end

    #For each specified state
    for (state_key, state_value) in agent.states
        #Add it to the history
        agent.history[state_key] = [state_value]
    end

    #Check agent for settings of shared parameters
    check_agent(agent)

    return agent
end


"""
Function for checking the structure of the agent
"""
function check_agent(agent::Agent)

    if length(agent.parameter_groups) > 0

        ## Check for the same derived parameter in multiple shared parameters 
        #Get out the derived parameters of all shared parameters 
        grouped_parameters = [
            parameter for list_of_grouped_parameters in [
                agent.parameter_groups[parameter_key].grouped_parameters for
                parameter_key in keys(agent.parameter_groups)
            ] for parameter in list_of_grouped_parameters
        ]

        #check for duplicate names
        if length(grouped_parameters) > length(unique(grouped_parameters))
            #Throw an error
            throw(
                ArgumentError(
                    "At least one parameter is set by multiple shared parameters. This is not supported.",
                ),
            )
        end
    end


end
