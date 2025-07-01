# # The Rescorla-Wagner

# ## Introduction
# The Rescorla-Wagner model is a classical model of associative learning, which describes how expectations about outcomes are updated based on observed outcomes.
# Originally, it was developed to model reinforcement learning tasks, where an organism learns to associate actions with various rewards.
# Since then, it has been applied in a variety of contexts, and is now a canonical model in reinforcement learning and cognitive psychology for modelling how agents learn from their environment.
# In ActionModels, the Rescorla-Wagner model is provided as a premade model. It functions as a perceptual model, and can be combined with various response models to make a full action model.
# In the following the Rescorla-Wagner model is described mathematically, and it's binary and categorical variants are introduced.
# In the next sections, we will demonstrate how it can be used with ActionModels 

# ### The continuous Rescorla-Wagner model
# The classic rescorla wagner model has a single changing state, the expected value $V_t$ of an outcome at time $t$.
# This expectation is updated by the Rescorla-Wagner update rule:

# $$V_t = V_{t-1} + \alpha (o_t - V_{t-1})$$

# where $o_t$ is the observed outcome at time $t$, and $\alpha$ is the learning rate.
# The continuous Rescorla-Wagner model therefore has a total of two parameters:
# - The initial expected value $V_0 \in \mathbb{R}$ (by default set to 0)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# And one state:
# - The expected value $V_t \in \mathbb{R}$, which is updated on each timestep

# The expected value $V_t$ can be used to determine the action in various ways, depending on the response model of choice. 
# A classic response model, which is the default in the ActionModels implementation, is a report with Gaussian noise.
# Here an observation is used to update the Rescorla-Wagner, and the expectation is then reported as the action, with some noise.
# The report action is sampled from a Gaussian distribution with the expected value $V_t$ as mean and a noise parameter $\beta$ as standard deviation:

# $$a_t \sim \mathcal{N}(V_t, \beta)$$

# This report action model then has one additional parameter:
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)
# Takes one observation:
# - The observation $o_t \in \mathbb{R}$
# And returns one action:
# - The report action $a_t \in \mathbb{R}$, which is sampled from the Gaussian distribution described above.

# ### The binary Rescorla-Wagner model
# In its binary variant, the Rescorla-Wagner model receives a binary observation (as opposed to a continuous one), and its task is to learn the probability of the binary outcome.
# Here, the expected value $V_t$ is transformed with a logistic or sigmoid function to ensure it is between 0 and 1, before it is used to calculate the prediction error:

# $$V_t = V_{t-1} + \alpha (o_t - \sigma(V_{t-1}))$$

# where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic function.
# The binary rescorla Wagner thus has a total of two parameters:
# - The initial expected value $V_0 \in \mathbb{R}$ (by default set to 0)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# And one state:
# - The expected value $V_t \in \mathbb{R}$

# A classic response model, which is the default in the ActionModels implementation, is a binary report with Bernoulli noise.
# Here an observation is used to update the Rescorla-Wagner, and the expectation is then reported as the action, with some noise.
# The binary report action is sampled from a Bernoulli distribution with the sigmoid-transformed expected value $V_t$ as probability. An action precision (the inverse action noise $\beta$) is used to control the noise of the action:

# $$a_t \sim \text{Bernoulli}(\sigma(V_t * \beta^{-1}))$$

# This report action model then has one additional parameter:
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)
# Takes one observation:
# - The binary observation $o_t \in \{0, 1\}$
# And returns one action:
# - The report action $a_t \in \{0, 1\}$, which is sampled from the Bernoulli distribution described above.

# ### The categorical Rescorla-Wagner model
# In its categorical variant, the Rescorla-Wagner model receives a categorical observation (as opposed to a continuous one), and its task is to learn the probability of each observation category occuring.
# Here the observation is transformed into a one-hot encoded vector, representing for each category whether it was observed or not.
# For each category, the expected value $V_t$ is updated with the binary Rescorla-Wagner update rule:

# $$V_{t, c} = V_{t-1, c} + \alpha (o_{t, c} - \sigma(V_{t-1, c}))$$

# where $o_{t, c}$ is the observation for category $c$ at time $t$, and 

# $\sigma(x) = \frac{1}{1 + e^{-x}}$ 

# is the logistic function.
# The categorical Rescorla-Wagner model therefore has a total of two parameters:
# - The initial expected values $V_0 \in \mathbb{R}$, which is a vector of values for each category (by default set to a vector of zeros)
# - The learning rate $\alpha \in [0,1]$ (by default set to 0.1)
# And one state:
# - The expected values $V_t \in \mathbb{R}$, which is a vector of expected values for each category, updated on each timestep

# A classic response model, which is the default in the ActionModels implementation, is a categorical report with noise.
# Here a categorical observation is used to update the Rescorla-Wagner, and the category with the highest expected probability is then reported as the action, with some noise.
# The categorical report action is sampled from a Categorical distribution with the softmax-transformed expected values $V_t$ as probabilities, weighted by an action precision (the inverse action noise $\beta$).

# $$a_t \sim \text{Categorical}(σ(V_t * \beta^{-1}))$$ 

# where $\sigma(x)$ is the softmax function 

# $σ(x) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$

# for each category $i$, which ensures that the probabilities sum to 1.
# This report action model then has one additional parameter:
# - The action noise $\beta \in [0, \infty]$ (by default set to 1)
# Takes one observation:
# - The categorical observation $o_t \in \{1, 2, \ldots, n\}$, which is the category observed at time $t$
# And returns one action:
# - The report action $a_t \in \{1, 2, \ldots, n\}$, which is sampled from the Categorical distribution described above.


# ## The premade Rescorla-Wagner constructor
# In this section, we demonstrate how to use the premade Rescorla-Wagner model constructor provided by ActionModels.
# This can construct continuous, binary and categorical Rescorla-Wagner model variants, as described above.
# It can be used with the standard report actions above, or with custom response models.

# First we load ActionModels, as well as StatsPlots for plotting the results:
using ActionModels
using StatsPlots

# We can then create the configuration struct for the Rescorla-Wagner model.
rescorla_wagner_config = RescorlaWagner();

# And use that to create an action model.
action_model = ActionModel(rescorla_wagner_config)

# Which can now be used to simulate or fit the model as usual.
# By default, the RescorlaWagner constructor creates a continuous Rescorla-Wagner model with a Gaussian report action.
# It can also create binary and categorical Rescorla-Wagner models, by specifying the `type` keyword argument.
# Finally, it is possible to specify whether the expected value should be updated before or after the action is sampled, by setting the `act_before_update` keyword argument.
# This is useful with datasets that are structured so that the observation for a given timestep is not available until after the action has been sampled, such as in some reinforcement learning tasks.

continuous_RW_config = RescorlaWagner(
    type = :continuous,
    initial_value = 0.0,        # Initial expected value
    learning_rate = 0.1,        # Learning rate
    action_noise = 1.0,         # Action noise (only applicable with default report action)
    act_before_update = false,  # Whether to act before updating the expected value
)
continuous_RW = ActionModel(continuous_RW_config)

binary_RW_config = RescorlaWagner(
    type = :binary,
    initial_value = 0.0,        # Initial expected value
    learning_rate = 0.1,        # Learning rate
    action_noise = 1.0,         # Action noise (only applicable with default report action)
    act_before_update = false,  # Whether to act before updating the expected value
)
binary_RW = ActionModel(binary_RW_config)

categorical_RW_config = RescorlaWagner(
    type = :categorical,
    n_categories = 4,           # Number of categories
    initial_value = zeros(4),   # Initial expected values for each category - should be equal to the number of categories
    learning_rate = 0.1,        # Learning rate
    action_noise = 1.0,         # Action noise (only applicable with default report action)
    act_before_update = false,  # Whether to act before updating the expected value
)
categorical_RW = ActionModel(categorical_RW_config);

# This means that users do not have to create an action model manually, but can proceed directly to simulating or fitting the model.
# With a categorical Rescorla-Wagner model, it looks like this (first simulating and thenf itting a single session):

#Load packages
using ActionModels
using StatsPlots

#Create action model
action_model = ActionModel(RescorlaWagner(type = :categorical, n_categories = 4))

#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [2, 1, 2, 2, 1, 2, 3, 4, 3, 2, 2, 2, 1, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = MvNormal(zeros(4), I),
    ),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

# And we can see that the parameters have been succesfully estimated. The initial_value has an index for each session, and for each category.
# This action model can be used normally as described in the rest of the documentation. 

# ### Custom response models
# In the examples above, we have used the default report action with the rescorla-wagner model.
# The precreated model, however, also allows for using custom response models.
# Here we show how to create the Gaussian report action as a custom response model - this is the default response model when using the continuous Rescorla-Wagner model.
# We will make a minor change, and add a bias parameter to the response model, which will be added to the expected value before sampling the action.

# The response model itself should be a function that takes the model attributes as input and returns a distribution.
# The Rescorla-Wagner will have either been updated or not, depending on the `act_before_update` keyword, so the response model should be able to handle either case.
response_model = function biased_gaussian_report(attributes::ModelAttributes)
    rescorla_wagner = attributes.submodel
    Vₜ = rescorla_wagner.expected_value
    β = load_parameters(attributes).action_noise
    b = load_parameters(attributes).bias
    return Normal(Vₜ + b, β)
end;

# In addition to the response model function, we need to specify the parameters, observations, and actions that the response model uses.
# This uses the usual syntax for creating action models.
response_model_parameters = (; action_noise = Parameter(1.0), bias = Parameter(0.0));
response_model_observations = (; observation = Observation(Float64));
response_model_actions = (; report = Action(Normal));

# We can now create the Rescorla-Wagner action model with the custom response model.
action_model = ActionModel(
    RescorlaWagner(
        type = :continuous,
        response_model = response_model,
        response_model_parameters = response_model_parameters,
        response_model_observations = response_model_observations,
        response_model_actions = response_model_actions,
    ),
)

# And continue to simulate with the biased report response model.
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Set a strong bias but low noise
set_parameters!(agent, (; bias = 2.0, action_noise = 0.1))

#Create observation
observations = collect(0:0.1:2) .+ randn(21) * 0.1

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Plot the trajectory
plot(agent, :expected_value)
plot!(simulated_actions, seriestype = :scatter, label = "Actions")
plot!(observations, seriestype = :scatter, label = "Observations")
title!("Biased actions relative to the expected value")

# We can also fit the the model to estimate the action bias

model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        bias = Normal(),
    ),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

plot(chns)

# And we can see that the bias parameter has been estimated correctly, as has the other parameters.

# ### Multiple observation sources
# In some experiments, there are multiple soruces of observations. It is then a common strategy to use a single Rescorla-Wagner model to learn the expected value of each observation source, and then combine these expected values in a custom response model.
#TODO: This needs multiple submodels for implementation



# ## The RescorlaWagner submodel
# In the previous section, we demonstrated how to use the premade Rescorla-Wagner model constructor provided by ActionModels.
# ActionModels also provides a RescorlaWagner submodel, which can be used to create custom action models with the Rescorla-Wagner updates as a submodel.
# In this section, we will demonstrate how to use the RescorlaWagner submodel to create custom action models, which can be useful in more complex scenarios.
# This also serves as a demonstration of how the premade Rescorla-Wagner model is implemented in ActionModels.

# In the following, we will demonstrate each of the three Rescorla-Wagner variants: continuous, binary and categorical.
# We will in each case combine them with the standard report action described above, but they can also be combined with other response models.

# First we load the ActionModels package, as well as StatsPlots for plotting the results:
using ActionModels
using StatsPlots

# ### Continuous variant
# We can then create a Rescorla-Wagner submodel with the Rescorla-Wagner constructor.
submodel = ActionModels.ContinuousRescorlaWagner(
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.1,  # Learning rate
);

# And an accompanying action model function with a Gaussian report action
model_function = function rescorla_wagner_gaussian_report(
    attributes::ModelAttributes,
    observation::Float64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the expected value from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Return the action distribution, which is a Gaussian with the expected value Vₜ as mean and the action noise β as standard deviation
    action_distribution = Normal(Vₜ, β)

    return action_distribution
end;

# We can then create an `ActionModel` object
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation()),
    actions = (; report = Action(Normal)),
)

# And proceed to simulate
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Set parameters
set_parameters!(agent, (; action_noise = 0.1, learning_rate = 0.5))

#Create observations
observations = collect(0:0.1:2) .+ randn(21) * 0.1

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Plot trajectory of expected values
plot(agent, :expected_value)
plot!(simulated_actions, seriestype = :scatter, label = "Actions")
plot!(observations, seriestype = :scatter, label = "Observations")
title!("Expectation trajectory and sampled actions", legend = :topright)

# Or fit the model

#Fit the model to the simulated actions
model = create_model(
    action_model,
    (learning_rate = LogitNormal(), action_noise = LogNormal(), initial_value = Normal()),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

plot(chns)

# ### Binary variant
# We can also create a binary Rescorla-Wagner submodel:
submodel = ActionModels.BinaryRescorlaWagner(
    initial_value = 0.0,  # Initial expected value
    learning_rate = 0.5,  # Learning rate
);

# And an accompanying action model function with a Bernoulli report action.
# Now, the observation is a binary value (0 or 1), and the expected value is transformed with a sigmoid function before being used to calculate the action distribution.
model_function = function rescorla_wagner_bernoulli_report(
    attributes::ModelAttributes,
    observation::Int64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Transform the action noise into a precision (or inverse noise)
    β⁻¹ = 1 / β

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the binary observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the expected value from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Transform the expected value with a logistic function to get the action probability, weighted by the action precision β⁻¹
    action_probability = logistic(Vₜ * β⁻¹)

    #Return the action distribution, which is a Bernoulli distribution with the action probability as parameter
    action_distribution = Bernoulli(action_probability)

    return action_distribution
end;

# We can then create an `ActionModel` object, with the observation as an integer (0 or 1) and the action as a Bernoulli distribution.
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation(Int64)),
    actions = (; report = Action(Bernoulli)),
)

# And proceed to simulate
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Get the history of expected values and transform them to probability space
transformed_expected_values = logistic.(get_history(agent, :expected_value))

#Plot the trajectory of expected values (starting at time 0, before receiving inputs)
plot(0:length(transformed_expected_values)-1, transformed_expected_values, label = "Expected value (probability space)")
plot!(simulated_actions, seriestype = :scatter, label = "Actions")
plot!(observations, seriestype = :scatter, label = "Observations")

# Or to fit the model

model = create_model(
    action_model,
    (learning_rate = LogitNormal(), action_noise = LogNormal(), initial_value = Normal()),
    observations,
    Int64.(simulated_actions),
)

chns = sample_posterior!(model)

plot(chns)

# ### Categorical variant
# We can also create a categorical Rescorla-Wagner submodel:
submodel = ActionModels.CategoricalRescorlaWagner(
    n_categories = 4,           # Number of categories
    initial_value = zeros(4),   # Initial expected values for each category
    learning_rate = 0.5,        # Learning rate
);

# And an accompanying action model function with a categorical report action.
# Now, the observation is a categorical value (1, 2, 3 or 4), and the expected value is transformed with a softmax function before being used to calculate the action distribution.
# We additionally rely on LogExpFunctions for the softmax function.
using LogExpFunctions

model_function = function rescorla_wagner_categorical_report(
    attributes::ModelAttributes,
    observation::Int64,
)
    #Load the action noise parameter
    parameters = load_parameters(attributes)
    β = parameters.action_noise

    #Transform the action noise into a precision (or inverse noise)
    β⁻¹ = 1 / β

    #Extract Rescorla-Wagner submodel
    rescorla_wagner = attributes.submodel

    #Update the Rescorla-Wagner expectation based on the categorical observation
    ActionModels.update!(rescorla_wagner, observation)

    #Extract the vector of expected values from the Rescorla-Wagner submodel
    Vₜ = rescorla_wagner.expected_value

    #Transform the expected value with a logistic function to get the action probability, weighted by the action precision β⁻¹
    action_probabilities = softmax(Vₜ .* β⁻¹)

    #Return the action distribution, which is a Bernoulli distribution with the action probability as parameter
    action_distribution = Categorical(action_probabilities)

    return action_distribution
end;

# We can then create an `ActionModel` object, with the observation as an integer (0 or 1) and the action as a Bernoulli distribution.
action_model = ActionModel(
    model_function,
    submodel = submodel,
    parameters = (; action_noise = Parameter(1.0)),
    observations = (; observation = Observation(Int64)),
    actions = (; report = Action(Categorical)),
)

# And proceed to simulate:
#Initialize agent
agent = init_agent(action_model, save_history = :expected_value)

#Create observation
observations = [2, 1, 2, 2, 1, 2, 3, 4, 3, 2, 2, 2, 1, 1]

#Simulate actions
simulated_actions = simulate!(agent, observations)

#Get number of timesteps
n_timesteps = length(observations)

# Get the history of expected values, which is a vector of vectors for each category
expected_values = get_history(agent, :expected_value)

#Collect the expected values for each category in a matrix, and transform them to probability space
expected_values = hcat(
    logistic.([expected_values[i][1] for i = 1:length(expected_values)]),
    logistic.([expected_values[i][2] for i = 1:length(expected_values)]),
    logistic.([expected_values[i][3] for i = 1:length(expected_values)]),
    logistic.([expected_values[i][4] for i = 1:length(expected_values)]))

#Use a softmax to normalize the expected values for each category into a proper categorical distribution
expected_values = transpose(hcat(softmax.(eachrow(expected_values))...))

# Create plot title
plot_title = plot(
    title = "Estimated probability of each category over time",
    grid = false,
    showaxis = false,
    bottom_margin = -180Plots.px,
)
#Create plot of probabilities changing
prob_plot = plot(0:n_timesteps, expected_values, label = ["Cat 1" "Cat 2" "Cat 3" "Cat 4"])
#Create plot of observations and simulated actions
obs_plot = plot(observations, seriestype = :scatter, label = "Observed category")
plot!(simulated_actions .+ 0.1, seriestype = :scatter, label = "Prediction for next timestep")  
#Create full plot
plot(plot_title, prob_plot, obs_plot, layout = (3, 1))

# Or to fit the model
model = create_model(
    action_model,
    (
        learning_rate = LogitNormal(),
        action_noise = LogNormal(),
        initial_value = MvNormal(zeros(4), I),
    ),
    observations,
    simulated_actions,
)

chns = sample_posterior!(model)

plot(chns)

