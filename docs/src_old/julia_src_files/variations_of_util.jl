# # # Variations of utility functions

# # This section contains a list of convenient utility functions. Many of these are used throughout the documentation and you can find all here.


# # Content
# #   - [getting states of an agent with: get_states()](#Getting-States)
# #   - [getting parameters of an agent with: get_parameters()](#Getting-Parameters)
# #   - [setting parameters of an agent with: set_parameters()](#Setting-Parameters)
# #   - [getting history of an agent with: get_history()](#Getting-History)
# #   - [resetting an agent with reset!() and getting posteriors with get_posteriors()](#Getting-Posteriors)

# # We will define an agent to use during demonstrations of the utility functions:
# using ActionModels #hide

# agent = premade_agent("binary_rescorla_wagner_softmax")

# # ## Getting States
# # The get_states() function can give you a single state, multiple states and all states of an agent. 

# # Let's start with all states
# get_states(agent)

# # Get a single state
# get_states(agent, "value")

# # Get multiple states
# get_states(agent, ["value", "action"])


# # ## Getting Parameters

# # get\_parameters() work just like get_states, but will give you the parameters of the agent:

# # lets start with all parameters
# get_parameters(agent)

# # Get a single parameter
# get_parameters(agent, ("initial", "value"))

# # Get multiple parameters
# get_parameters(agent, [("initial", "value"), "learning_rate"])


# # ## Setting Parameters

# # Setting a single parameter in an agent
# set_parameters!(agent, ("initial", "value"), 1)

# # Setting multiple parameters in an agent
# set_parameters!(agent, Dict("learning_rate" => 3, "action_precision" => 0.5))

# # See the parameters we have set uising get_parameters function
# get_parameters(agent)


# # ## Getting History 

# # To get the history we need to give inputs to the agent. Let's start by giving a single input

# give_inputs!(agent, 1)

# # We can now get the history of the agent's states. We can have a look at the "value" state.

# get_history(agent, "value")

# # Get multiple states' histories

# get_history(agent, ["value", "action"])

# # Lastly, get all history of the agent

# get_history(agent)


# # ## Getting Posteriors 

# # get\_posteirors() is a funcion for extracting parameters from a Turing chain. Let us set up a fitted model:

# # Let us reset our agent and make it ready for new input
# reset!(agent)

# # Define a range of inputs
# inputs = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

# # Define actions
# actions = give_inputs!(agent, inputs)

# # Set a prior for the parameter we wish to fit
# using Distributions
# priors = Dict("action_precision" => Normal(1, 0.5), "learning_rate" => Normal(1, 0.1))

# # Fit the model
# fitted_model = fit_model(agent, priors, inputs, actions)


# # We can now use the get_posteriors() 
# get_posteriors(fitted_model)
