
# # # Building your own action model

# # In this section we will demonstrate the following

# #   - [A structure comparison between an agent and an action model](#visual-structure-of-agent-and-action-model)
# #   - [Tutorial on how to make the Rescorla-Wagner softmax action model](#Building-the-Rescorla-Wagner-action-model)
# #   - [Creating a custom agent to the action model with init_agent()](#Create-An-Agent)

# # An action model should always return an action distribution from which an aciton can be sampled.

# # ## visual structure of agent and action model 

# # ![Image1](../images/actionmodel_custom.png)


# # let us start by the left side of the model. We have an agent with a set of defined parameters, states and a setting. 

# # When initializing an action model you can start by reading in the agent's parameters. It can depend on the action model how many states you use during the update step. In this example we only use one state (which is the one that has an initial state parameter).

# # During the update step we compute the remaining two states in the agent. State\_2 and state\_3 are outputs of the update step which are saved in the agent's history. In our example structure you can see that the updated\_state\_3 is the parameter in our action distribution.

# # When building a custom action model you will have to (roughly) think of 3 things:

# #   - What type of action do I want to simulate? 
# #   - What does the update step constist of?
# #   - What variables (parameters and states) are used in the update step and which do I wish to save in the agent history?

# # ## Building the Rescorla-Wagner action model

# # In this example we will construct an action model according to Rescorla-Wagner. First, let us recall the parameters and states needed for this action model. 

# # We need two parameters: learning rate and softmax action precision. These parameters are specified when defining the agent. The main state for the update step is the "input value" state, and this is the only value we need to have in order to calculate an action.

# # It can be nice to save other elements from the action model as states for analysis and plotting. We would in this case like to save the action probability and the transformed value in probability space. These two states don't influence the model in new trials, but are just extra output from the action model.

# # Let us build our action model. The structure of the action model is equal to the strucutre in the figure.

# # First, let us define the function name and which input it takes:
# using ActionModels
# using Distributions
# using Plots, StatsPlots

# # In the next section you will be introduced to premade agents and models.

# # Rescorla Wagner continuous


# function continuous_rescorla_wagner_gaussian(agent::Agent, input::Real)

#     ## Read in parameters from the agent
#     learning_rate = agent.parameters["learning_rate"]
#     action_noise = agent.parameters["action_noise"]

#     ## Read in states with an initial value
#     old_value = agent.states["value"]

#     ##We dont have any settings in this model. If we had, we would read them in as well.
#     ##-----This is where the update step starts -------

#     ##Get new value state
#     new_value = old_value + learning_rate * (input - old_value)


#     ##-----This is where the update step ends -------
#     ##Create Bernoulli normal distribution our action probability which we calculated in the update step
#     action_distribution = Distributions.Normal(new_value, action_noise)

#     ##Update the states and save them to agent's history
#     update_states!(agent, Dict("value" => new_value, "input" => input))

#     ## return the action distribution to sample actions from
#     return action_distribution
# end


# #-- define parameters and states --#
# parameters =
#     Dict("learning_rate" => 0.8, "action_noise" => 1, InitialStateParameter("value") => 0)

# states = Dict("value" => missing, "input" => missing)

# #-- create agent --#
# agent = init_agent(
#     continuous_rescorla_wagner_gaussian,
#     parameters = parameters,
#     states = states,
# )


# #-- define observations --#
# inputs = [1, 2, 3, 4, 6, 4, 10, 2, 1]

# #-- give them to the agent --#
# actions = give_inputs!(agent, inputs)

# #Minor detail for plotting reasons
# inputs = append!(Float64.([0]), inputs)
# actions = append!(Float64.([0]), actions)

# #Plot
# plot(inputs, linetype = :scatter, label = "input")
# plot_trajectory!(agent, "value", linetype = :line)
# plot!(actions, linetype = :scatter, label = "action")




# # ## A Binary Rescorla-Wagner
# function binary_rescorla_wagner_softmax(agent::Agent, input::Union{Bool,Integer})

#     #Read in parameters
#     learning_rate = agent.parameters["learning_rate"]
#     action_precision = agent.parameters["action_precision"]

#     #Read in states
#     old_value = agent.states["value"]

#     #Sigmoid transform the value
#     old_value_probability = 1 / (1 + exp(-old_value))

#     #Get new value state
#     new_value = old_value + learning_rate * (input - old_value_probability)

#     #Pass through softmax to get action probability
#     action_probability = 1 / (1 + exp(-action_precision * new_value))

#     #Create Bernoulli normal distribution with mean of the target value and a standard deviation from parameters
#     action_distribution = Distributions.Bernoulli(action_probability)

#     #Update states
#     update_states!(
#         agent,
#         Dict(
#             "value" => new_value,
#             "value_probability" => 1 / (1 + exp(-new_value)),
#             "action_probability" => action_probability,
#             "input" => input,
#         ),
#     )

#     return action_distribution
# end



# # ## Create An Agent


# # We can define the agent now. Let us do it with the init_agent() function. We need to define parameters, states and action model. 

# #Set the parameters:

# parameters =
#     Dict("learning_rate" => 1, "action_precision" => 1, InitialStateParameter("value") => 0)

# # We set the initial state parameter for "value" state because we need a starting value in the update step. 

# # Let us define the states in the agent:
# states = Dict(
#     "value" => missing,
#     "value_probability" => missing,
#     "action_probability" => missing,
#     "input" => missing,
# )

# # And lastly the action model:
# action_model = binary_rescorla_wagner_softmax

# # We can now initialize our agent with the action model, parameters and states.
# agent = init_agent(action_model, parameters = parameters, states = states)

# inputs = [1, 0, 0, 0, 1, 1, 0, 1, 0]

# give_inputs!(agent, inputs)

# plot_trajectory(agent, "action")
