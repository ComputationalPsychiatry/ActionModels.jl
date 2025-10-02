

# # # Welcome to the ActionModels.jl package!

# # ActionModels.jl is a powerfull and novel package for computational modelling of behavior and cognition. The package is developed with a intention to make computaitonal modelling intuitive, fast and easily adaptive to your experimental and simulation needs. 

# # With ActionModels.jl you can define a fully customizable behavioral model and easily fit them to experimental data and used for computational modelling.

# # we provide a consice introduction to this framework for computational modelling of behvior and cognition and its accompanying terminology.

# # After this introduction, you will be presented with a detailed step-by-step guide on how to use ActionModels.jl to make your computational model runway-ready.


# # ## Getting started 

# # Defning a premade agent
# using ActionModels

# # Find premade agent, and define agent with default parameters
# premade_agent("help")

# agent = premade_agent("binary_rescorla_wagner_softmax")

# # Set inputs and give inputs to agent

# inputs = [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
# actions = give_inputs!(agent, inputs)

# using StatsPlots
# plot_trajectory(agent, "action_probability")

# # Fit learning rate. Start by setting prior

# using Distributions
# priors = Dict("learning_rate" => Normal(0.5, 0.5))

# # Run model
# # chains = fit_model(agent, priors, inputs, actions, n_chains = 1, n_iterations = 10)

# # Plot prior and posterior
# # plot_parameter_distribution(chains, priors)

# # # Get posteriors from chains

# # get_posteriors(chains)


# # ## Computaitonal modelling of behavior

# # In computational behavioral modelling the aim is to produce mechanistic models of the processes underlying observed behavior. Importantly, to validate and test models, or to use them to distinguish groups of individuals, models must be fitted to emprircal data, an often complicated process that this package is designed to make as simple as possible. 

# # You can imagine an experimental setting where a participant is to press one of two buttons. One of the buttons elict a reward and the other elict a small electrical chock. The experimenter has set up a system, so that the probability of choosing the right button shifts over time. You can imagine this as being 80/20 on left being reward-button in the first 10 trials, and then the probability shifts to 20/80 on left being rewarding in the last 10 trials. 

# # The participant wishes to press the reward button and avoid the electrical chock. During the first 10 trials the participant would, through feedback, alter their behavior to select the left button the most due to the reward distribution between buttons. In the last 10 trials the preffered button press would swich. 

# # After the experiement the hypothetical data could be the participants button choice. We can model the behvaior of the participant in the experiment with an agent and an action model. We can with the action model approximate how information is processed in the participant (in modelling the participant becomes the agent), and how the agent produces actions with a specific model. We try to create an artificially mapping between input and action that can explain a persons behavior

# # The agent contains a set of "states" and "parameters". The parameters in an agent are analogous to some preconcieved belief of the agent that can't be changed during the experiment. A very caricatured example could be if the participant had a strong color preference for one of the buttons which influences their decisions on a constant level. When we model behvaior and set these parameters they are related to some theoretically grounded elements from the action model. We will later build an action model from scratch where real parameters will show up.

# # The states in an agent change over time, and the way they change depend on the action model. This structure will be elaborated more on in the next section where we go into depth with agents and action models.
