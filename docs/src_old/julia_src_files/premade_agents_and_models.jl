
# # # Creating your agent using premade agents and models in the ActionModels package

# # In this section we will demonstrate:
# #    - [Defining an agent with the premade_agent() function](#Defining-an-agent-with-premade_agent())
# #    - [Get an overview of the parameters and states](#Get-an-overview-of-the-parameters-and-states-in-a-premade-agent)
# #    - [Set one and more parameter values in an agent](#Set-one-and-more-parameter-values-in-an-agent)
# #    - [Defining an agent with custom parameter values](#Defining-an-agent-with-custom-parameter-values)

# # ## Defining an agent with premade_agent()


# using ActionModels #hide

# # you can define a premade agent using the premade_agent() function. The function call is the following:

# #premade_agent(model_name::String, config::Dict = Dict(); verbose::Bool = true)

# # Model_name is the type of premade agent you wish to use. You can get a list of premade agents with the command:

# premade_agent("help")

# # Lets create an agent. We will use the "premade\_binary\_rw\_softmax" agent with the "binary\_rw\_softmax" action model. You define a default premade agent with the syntax below:
# agent = premade_agent("binary_rescorla_wagner_softmax")

# # In the Actionmodels.jl package, an agent struct consists of the action model name (which can be premade or custom), parameters and states.  
# # The premade agents are initialized with a set of configurations for parameters, states and initial state parameters.


# # ## Get an overview of the parameters and states in a premade agent 

# # Let us have a look at the parameters in our predefined agent with the fucntion get_parameters

# get_parameters(agent)

# # If you wish to get one of the parameters, you can change the method of the function by specifying the parameter you wish to retrieve:

# get_parameters(agent, "learning_rate")

# # Now, let us have a look at the states of the agent with the get_states() function.

# get_states(agent)

# # the "value" state is initialized with an initial state parameter (this is the ("initial", "value") parameter seen in the get_parameters(agent) call). The other states are missing due to the fact, that the agent has not recieved any inputs. 
# # You can get specific states by inputting the state you wish to retrieve. This will get more interesting once we give the agents inputs.

# get_states(agent, "value")


# # ## Set one and more parameter values in an agent

# # As mentioned above, the premade agent is equipped with dedault parameter values. If you wish to change these values you can do it in different ways.
# # We can change one of the parameters in our agent like below by specifying parameter and the value to set the parameter to.

# set_parameters!(agent, "learning_rate", 1)

# get_parameters(agent)

# # If you wish to change multiple parameters in the agent, you can define a dict of parameters folowed by the value like this

# set_parameters!(
#     agent,
#     Dict("learning_rate" => 0.79, "action_precision" => 0.60, ("initial", "value") => 1),
# )

# # ## Defining an agent with custom parameter values

# # If you know which parameter values you wish to use when defining your agent, you can specify them in the beginning as a dict() with parameter name as a string followed by the value.
# agent_custom_parameters = premade_agent(
#     "binary_rescorla_wagner_softmax",
#     Dict("learning_rate" => 0.7, "action_precision" => 0.8, ("initial", "value") => 1),
# )

# #and we can retrieve the new parameters with the get_parameters() function

# get_parameters(agent_custom_parameters)
