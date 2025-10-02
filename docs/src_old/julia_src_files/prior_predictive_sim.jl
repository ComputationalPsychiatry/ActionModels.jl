
# # # Predictive Simulations

# # In order to set a good prior, a prior predicitve simulation is a principled method for testing 

# # If you wish to fit a certain parameter in your agent, you can use the prior predictive simulation to simulate different states e.g. actions with parameter values sampled from this prior.


# # Overview

# # -  [What prior/posterior predictive simulation is](#Predictive-simulations)
# #  - [Introduction to the plot\_predictive\_simulation() function](#The-plot\_predictive\_simulation())
# #  - [example of prior predictive simulation](#Example-of-prior-predictive-simulaiton)
# #  - [example of posterior predictive simulation](#Posterior-predictive-simulations)



# # ## Predictive simulations

# # When we do prior predictive simulations, it is often with the goal of setting a reasonable prior based on our expectations on what range our target state might be in. An example could be, that we want to fit a target state influencing actions measured in reaction time. Our prior for this parameter should not make the agent produce negative reaction times during the simulation. 
# # Let us take a look at the figure illustrating the process:

# # ![Image1](../images/predictive_sim.png)

# # We have a prior of one (or more) target parameters which we want to fit, and take a random sample from the prior distribution. This prior is placed in the agent alongside with earlier specified fixed parameters. We simulate forward and give the agent inputs. Depending on which target state you are interested in, you get the history of that state. 

# # When the agent has been given all inputs it is reset to default, a new sample from the prior is drawn and the process is repeated the amount of times you have set the n_samples to.

# # It is possible to insert a fitted model as your posterior as well. With the exact same method (we simply sample from the posterior instead of prior) you can see how well the fitted parameters perform in retrieving plausable target states.

# # ## The plot\_predictive\_simulation()

# # The inputs to the function are the following

# # ![Image1](../images/plot_predictive_code.png)

# # ## Example of prior predictive simulaiton
# # Let us go through a prior predictive simulation

# #load packages
# using ActionModels

# #Define an agent
# agent = premade_agent("binary_rescorla_wagner_softmax")
# #Define input
# inputs = [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]

# #see which taget state wish to plot
# get_states(agent)
# #we choose action probability
# target_state = "action_probability"

# #see which parameter we wish to simulate from
# get_parameters(agent)

# #Let us choose leanring rate, and set a prior
# using Distributions
# prior_learning_rate = Dict("learning_rate" => Normal(1.2, 0.5))


# # Insert values in the function
# using Plots
# using StatsPlots
# plot_predictive_simulation(prior_learning_rate, agent, inputs, target_state)


# # The red dot in the plot shows the median simulated value for action probability from each run on the inputs with different parameter samples.


# # ## Posterior predictive simulations

# # To produce a posterior predictive simulation plot we need a posterior from a fitted model. To see in depth how to fit an agent model to data using fit\_model(), see section [INSERT LINK]
# # We will fit the learning rate using the prior we set earlier.  The only thing we need to fit, is actions. Let's simulate actions with give\_inputs()

# actions = give_inputs!(agent, inputs)

# fitted_model = fit_model(agent, prior_learning_rate, inputs, actions)

# # The plot\_predictive\_simulation() function recognizes a fitted model, more specifically a turing chain of posteriors. We can therefore input the fitted model as our posterior.

# # Insert the fitted model (with a posterior for learnig rate) and plot action probability as our target state.
# plot_predictive_simulation(fitted_model, agent, inputs, target_state)
