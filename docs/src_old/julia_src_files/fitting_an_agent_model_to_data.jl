# # # Fitting an agent model to data

# # The core of behavioral and cognitive computational modelling is to fit our models to data. One of the many essential features in the ActionModels.jl package is to fit your data with high speed and good stable performance.


# # In this section the following will be demonstrated

# #   - [Recap of fitting models](#So-far-with-the-Actionmodels.jl-package)
# #   - [The fit_model() function](#The-fit_model()-function)
# #   - [tutorial of fitting one and more parameters](#Tutorial-of-basic-use-of-fit_model())
# #   - [Plotting posteriors](#Plotting-functions)


# # ## So far with the Actionmodels.jl package

# # During the tutorial you should be comfortable with the terms agents, actions, action models, states and parameters as well as how to simulate actions. 
# # We can deifne a premade agent or create a custom agent with different kinds of action models. You can change the parameters of the agents and simulate actions by giving them inputs. You might have tried out for yourself how simulated actions change depending on which parameter values you set in your agent. 

# # This can lead os to what is meant by "fitting". We will again reference the illustration of comparing simulation and fitting:

# # ![Image1](../images/fitting_vs_simulation.png)

# # When we fit, we know the actions and inputs. As we have seen earlier with different parameter settings for agents, these change their "behavior" and actions quite drastically . When we fit the parameters of a model, we try to find the parameter values which make that model most likely to produce observed actions. 
# # Finding good guesses to these parameter values can be usefull when examining differences between groups in experimental settings. 

# # When we fit one or more parameters we need to set priors to sample from. These priors are initial guesses to where the plausable parameter values could be. These priors can with great benefit be informed priors, where values from previous studies influence your choice of prior.

# # ## The fit_model() function

# # The fit_ model() function takes the following inputs:


# # ![Image1](../images/fit_model_image.png)


# # Let us run through the inputs to the function one by one. 

# # - agent::Agent: a specified agent created with either premade agent or init\_agent.
# # - param_priors::Dict: priors (written as distributions) for the parameters you wish to fit. e.g. priors = Dict("learning\_rate" => Uniform(0, 1))
# # - inputs:Array: array of inputs.
# # - actions::Array: array of actions.
# # - fixed_parameters::Dict = Dict(): fixed parameters if you wish to change the parameter settings of the parameters you dont fit
# # - sampler = NUTS(): specify the type of sampler. See Turing documentation for more details on sampler types.
# # - n_iterations = 1000: iterations pr. chain.
# # - n_chains = 1: amount of chains.
# # - verbose = true: set to false to hide warnings
# # - show\_sample\_rejections = false: if set to true, get a message every time a sample is rejected.
# # - impute\_missing\_actions = false : if true, include missing actions in the fitting process.


# # ## Tutorial of basic use of fit_model()

# # In the first part of the tutorial we will not change the sampling and parallellization configurations or the error information / misssing actions settings. We will only work with the agent information and priors configurations. See elaborated use of fit_model() for working with more configurations
# # Let us use a premade agent:
# using ActionModels
# using Distributions
# using StatsPlots

# agent = premade_agent("binary_rescorla_wagner_softmax")

# # Let's give the agent some input and simulate a set of actions:

# inputs = [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1]

# actions = give_inputs!(agent, inputs)

# # Let's take a look at what parameter in the agent we could fit: 

# get_parameters(agent)

# # We want to fit learning_rate. Let us define a prior for the parameter. We set this prior as a normal distribution with mean 1 and standard deviation 0.5. It is important to note that it is possible to fit multiple parameters at the same time. In that case you would simply add more priors in the dict.

# priors = Dict("learning_rate" => Normal(1, 0.5))

# # When we set a prior for a parameter it overwrites the parameter in the agent during the fitting process. The prior is not part of the agent, but is only used in the fit_model() function. 

# # We have our agent, inputs, actions and priors. This is what we need to fit.

# # fitted_model = fit_model(agent, priors, inputs, actions)

# # As output you are presented with the summary statistics. 

# # We can plot the chains and distribution of the two chains.

# # plot(fitted_model)



# # ### Plotting functions
# # For plotting the prior against the posterior use the plot\_parameter\_distribution function. 

# # The first argument in the fuction is the fitted model and the second are the priors. The plot is a vizuialisation comparing the fitted parameters compared to priors

# # plot_parameter_distribution(fitted_model, priors)


# # You can extract the posterior parameters from a Turing chain with the get_posteriors() function:
# # get_posteriors(fitted_model)


# # ## Fitting multiple parameters

# # By adding multiple parameter priors you can autimatically fit them with fit\_model. Let's say you also want to fit softmax\_action\_precision.

# # Add an extra prior in the Dict
# # multiple_priors =
# #     Dict("learning_rate" => Normal(1, 0.5), "action_precision" => Normal(0.8, 0.2))

# # multiple_fit = fit_model(agent, multiple_priors, inputs, actions)

# # # Plot the parameter distribution 
# # plot_parameter_distribution(multiple_fit, multiple_priors)

# # # Extract the posteriors from the Turing chain
# # get_posteriors(multiple_fit)
