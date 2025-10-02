# # # Simulating actions with an agent

# # We will in this section introduce action simulation with the Actionmodels.jl package. 

# # ### contents of this section 
# #   - [The give_inputs() function](#Giving-Inputs-To-Agent)
# #   - [Giving a single input to the agent and retrieving history](#Give-a-single-input)
# #   - [Resetting the agent to initial state values](#Reset-Agent)
# #   - [Give_inputs() with a vector of inputs](#Multiple-Inputs)
# #   - [Plot State Trajectories](#Plotting-Trajectories-of-states)

# # ## Giving Inputs To Agent

# # With the ActionModels package you can, once you have defined your agent, use the function give_inputs to simulate actions.

# #give_inputs(agent::Agent, inputs::Real) 

# # When you give inputs to the agent, it produces actions according to the action model it is defined with.

# # As can be seen in the figure below, when we know all parameter values, states and the inputs we can simulate actions

# # ![Image1](../images/fitting_vs_simulation.png)

# # The type of inputs you can give to your agent depends on the agent and the action it generates depends on the corresponding action model.

# # Let us define our agent and use the dedault parameter configurations
# using ActionModels

# agent = premade_agent("binary_rescorla_wagner_softmax")

# # ## Give a single input
# # we can now give the agent a single input with the give_inputs!() function. The inputs for the Rescorla-Wagner agent are binary, so we input the value 1. 
# give_inputs!(agent, 1)

# # The agent returns either "false" or "true" which translates to an action of either 0 or 1. Given that we have evolved the agent with one input the states of the agent are updated. Let's see how we recover the history and the states of the agent after one run on input. We can do this with the get_history() function. With get_history we can get one or more target states or get the history of all states.

# # Let us have a look at the history from all states:

# get_history(agent)

# # You can see in the "value" state contains two numbers. The first number is the initial state parameter which is set in the agent's configurations (see "Creating your agent" for more on the parameters and states). The second value in the "value" state is updated by the input.
# # The three other states are initialized with "missing" and evolve as we give it inputs. The states in the agent are updated according to which computations the action model does with the input. For more information on the Rescorla-Wagner action model, check out the [LINK TO CHAPTER]

# # ## Reset Agent
# # We would like to reset the agent to its default values with the reset!() function:

# reset!(agent)

# # As you can see below, we have cleared the history of the agent.

# get_history(agent)

# # ## Multiple Inputs
# # We will now define a sequence of inputs to the agent. 

# inputs = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]

# # Let's give the inputs to the agent, and see which type of actions it will generate based on the inputs

# actions = give_inputs!(agent, inputs)

# # We can in the same manner get the history of the agent's states. We will have a look at the action state:
# get_history(agent, "action_probability")

# # ## Plotting Trajectories of states

# # we can visualize the different types of states using the function:

# #plot_trajectory(agent::Agent, target_state::Union{String,Tuple}; kwargs...)

# # The default title when using plot_trajectory() is "state trajectory". This can be changed by adding a title-call as below. We can plot the actions and the action probability of the agent in two seperate plots:
# using Plots
# using StatsPlots

# plot_trajectory(agent, "action", title = "actions")

# # We can change the state to plotting the action_probability

# plot_trajectory(agent, "action_probability", title = "acton probability")

# # We can add a new linetype for the plot:

# plot_trajectory(agent, "action_probability", title = "acton probability", linetype = :path)

# # We can layer the plots by adding "!" at the function call. We can add the actions plot to prior action probability plot:

# plot_trajectory!(agent, "action", title = "action probability and action")


# # ## If your agent computes more actions 

# # If you wish to set up an agent who produces multiple actions, e.g. reaction time and a choice,
# # you can use the "multiple_actions()" function. When setting up this type of agent, you define the different action models you want to use for each one of the wanted actions. 
# # Currently in the ActionModels.jl package we have not yet predefined actionmodels for different actions. For multiple actions you should define your own action models (see the advanced usage for how to do this)

# # we define our two action models. A continuous and binary Rescorla Wagner:

# using ActionModels
# using Distributions
# # Binary Rescorla Wagner
# function custom_rescorla_wagner_softmax(agent, input)

#     ## Read in parameters from the agent
#     learning_rate = agent.parameters["learning_rate"]
#     action_precision = agent.parameters["action_precision"]

#     ## Read in states with an initial value
#     old_value = agent.states["value_binary"]

#     ##We dont have any settings in this model. If we had, we would read them in as well. 
#     ##-----This is where the update step starts ------- 

#     ## Sigmoid transform the value
#     old_value_probability = 1 / (1 + exp(-old_value))

#     ##Get new value state
#     new_value = old_value + learning_rate * (input - old_value_probability)

#     ##Pass through softmax to get action probability
#     action_probability = 1 / (1 + exp(-action_precision * new_value))

#     ##-----This is where the update step ends ------- 
#     ##Create Bernoulli normal distribution our action probability which we calculated in the update step
#     action_distributions = Distributions.Bernoulli(action_probability)

#     ##Update the states and save them to agent's history 

#     agent.states["value_binary"] = new_value
#     agent.states["transformed_value"] = 1 / (1 + exp(-new_value))
#     agent.states["action_probability"] = action_probability

#     push!(agent.history["value_binary"], new_value)
#     push!(agent.history["transformed_value"], 1 / (1 + exp(-new_value)))
#     push!(agent.history["action_probability"], action_probability)

#     ## return the action distribution to sample actions from
#     return action_distributions
# end


# # Continuous Rescorla Wagner
# function continuous_rescorla_wagner_softmax(agent, input)

#     ## Read in parameters from the agent
#     learning_rate = agent.parameters["learning_rate"]

#     ## Read in states with an initial value
#     old_value = agent.states["value_cont"]

#     ##We dont have any settings in this model. If we had, we would read them in as well. 
#     ##-----This is where the update step starts ------- 

#     ##Get new value state
#     new_value = old_value + learning_rate * (input - old_value)

#     ##-----This is where the update step ends ------- 
#     ##Create Bernoulli normal distribution our action probability which we calculated in the update step
#     action_distributions = Distributions.Normal(new_value, 0.3)

#     ##Update the states and save them to agent's history 

#     agent.states["value_cont"] = new_value
#     agent.states["input"] = input

#     push!(agent.history["value_cont"], new_value)
#     push!(agent.history["input"], input)

#     ## return the action distribution to sample actions from
#     return action_distributions
# end


# # Define an agent

# parameters = Dict(
#     "learning_rate" => 1,
#     "action_precision" => 1,
#     ("initial", "value_cont") => 0,
#     ("initial", "value_binary") => 0,
# )

# # We set the initial state parameter for "value" state because we need a starting value in the update step. 

# # Let us define the states in the agent:
# states = Dict(
#     "value_cont" => missing,
#     "value_binary" => missing,
#     "input" => missing,
#     "transformed_value" => missing,
#     "action_probability" => missing,
# )


# agent = init_agent(
#     [continuous_rescorla_wagner_softmax, custom_rescorla_wagner_softmax],
#     parameters = parameters,
#     states = states,
# )



# inputs = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]


# #multiple_actions(agent, inputs)
