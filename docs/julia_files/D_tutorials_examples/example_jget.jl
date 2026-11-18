# # Jumping Gaussian Estimation Task
# In this tutorial, we will fit a Rescorla-Wagner model to data from the Jumping Gaussian Estimation Task (JGET).
# In the JGET, participants observe continuous outcomes sampled from a Gaussian distribution which at some trials "jumps" to be centered somewhere else.
# Participants must predict the outcome of the next trial based on the previous outcomes.
# The data we will use is from a study on schizotypy (Mikus et al., 2025), where participants completed the JGET and also filled out the Peters Delusions Inventory (PDI).
# The PDI is a self-report questionnaire that measures delusional ideation, and we will use it as a predictor in our model.
# There are several session per participant in the dataset, which will be modeled as separate experimental sessions.
# Data is taken from https://github.com/nacemikus/jget-schizotypy, where more information can also be found.

# ## Loading data
# First, we load the ActionModels package. We also load CSV and Dataframes for loading the data, and StatsPlots for plotting the results.
using ActionModels
using CSV, DataFrames
using StatsPlots

# Then we load the JGET data, which is available in the docs/example_data/JGET folder.

ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide
#Trial-level data
JGET_data = CSV.read(
    joinpath(docs_path, "example_data", "JGET", "JGET_data_trial_preprocessed.csv"),
    DataFrame,
    missingstring = ["NaN", ""],
)
JGET_data = select(JGET_data, [:trials, :ID, :session, :outcome, :response, :confidence])

#Subject-level data
subject_data = CSV.read(
    joinpath(docs_path, "example_data", "JGET", "JGET_data_sub_preprocessed.csv"),
    DataFrame,
    missingstring = ["NaN", ""],
)
subject_data = select(subject_data, [:ID, :session, :pdi_total, :Age, :Gender, :Education])

#Join the data
JGET_data = innerjoin(JGET_data, subject_data, on = [:ID, :session])

#Remove ID's with missing actions and missing PDI scores
JGET_data = combine(
    groupby(JGET_data, [:ID, :session]),
    subdata -> any(ismissing, Matrix(subdata[!, [:response]])) ? DataFrame() : subdata,
)
JGET_data = combine(
    groupby(JGET_data, [:ID, :session]),
    subdata -> any(ismissing, Matrix(subdata[!, [:pdi_total]])) ? DataFrame() : subdata,
)
disallowmissing!(JGET_data, [:pdi_total, :response])

#Make the outcome into a Float64
JGET_data.outcome = Float64.(JGET_data.outcome)

show(JGET_data)


# For this example, we will subset the data to only include four subjects in total, across the three experimental sessions.
# This makes the runtime much shorter. Simply skip this step if you want to use the full dataset.

JGET_data = filter(row -> row[:ID] in [20, 40, 60, 70], JGET_data);

# ## Creating the model
# Then we construct the model to be fitted to the data.
# We will use a classic Rescorla-Wagner model with a Gaussian report action model.
# The Rescorla-Wagner model is a simple reinforcement learning model that updates the expected value of an action based on the observed outcome.
# The Gaussian report action model assumes that the agent reports a continuous value sampled from a Gaussian distribution, where the mean is the expected value of the action and the standard deviation is a noise parameter.
# There are two parameters in the action model: the learning rate $\alpha$ and the action noise $\beta$.
# See the section on the [Rescorla-Wagner model](./rescorla_wagner.md) for more details.

# We create the Rescorla-Wagner action model using the premade model from ActionModels.
action_model = ActionModel(RescorlaWagner())

# We then specify which column in the data corresponds to the action (the response) and which columns correspond to the input (the outcome).
# There are two columns which jointly specify the sessions: the ID of the participant and the experimental session number.
action_cols = :response
observation_cols = :outcome
session_cols = [:ID, :session];

# We then create the full model. We use a hierarchical regression model to predict the parameters of the Rescorla-Wagner model based on the PDI score.
# First we will set appropriate priors for the regression coefficients.
# For the action noise, the outcome of the regression will be exponentiated before it is used in the model, so pre-transformed outcomes around 3 (exp(3) ≈ 20) are among the most extreme values to be expected.
# For the learning rate, the outcome of the regression will be passed through a logistic function, so pre-transformed outcomes around around 5 (logistic(5) ≈ 0.993) are among the most extreme values to be expected.
# This means that we should limit priors to be fairly narrow, so that the linear regression does not go too far into inappropriate parameter space, which will increase the runtime of the fitting process.

regression_prior = RegressionPrior(
    β = [Normal(0, 0.3), Normal(0, 0.2), Normal(0, 0.5)],
    σ = truncated(Normal(0, 0.3), lower = 0),
)

plot(regression_prior.β[1], label = "Intercept")
plot!(regression_prior.β[2], label = "Effect of session number")
plot!(regression_prior.β[3], label = "Effect of PDI")
plot!(regression_prior.σ, label = "Random intercept std")
title!("Regression priors for the Rescorla-Wagner model")
xlabel!("Regression coefficient")
ylabel!("Density")

# We then create the population model, which consists of two regression models: one for the learning rate and one for the action noise.
# We can then also create the full model using the `create_model` function.
population_model = [
    Regression(
        @formula(learning_rate ~ 1 + pdi_total + session + (1 | ID)),
        logistic,
        regression_prior,
    ),
    Regression(
        @formula(action_noise ~ 1 + pdi_total + session + (1 | ID)),
        exp,
        regression_prior,
    ),
]

model = create_model(
    action_model,
    population_model,
    JGET_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

# ## Fitting the model
# We are now ready to fit the model to the data.
# For this model, we will use the ForwardDiff automatic differentiation backend, which is the default, and which is faster for this model than the reverse-mode backends that work with it.
# Additionally, to keep the runtime of this tutorial short, we will only fit two chains with 500 samples each.
# We also pass `MCMCThreads()` to fit two chains in parallel. This should take up to 5 minutes on a standard laptop.

## Set AD backend ##
using ADTypes: AutoForwardDiff
adtype = AutoForwardDiff();

## Fit model ##
chns = sample_posterior!(model, MCMCThreads(), n_chains = 2, n_samples = 500, adtype = adtype)

# We can now inspect the results of the fitting process.
# We can plot the posterior distributions of the beta parameter for PDI's effect on the action model parameters.
# Here we can see that there may be a weak indication of an increase in action noise with increasing PDI score, although without the full dataset, the posterior is not very informative.
title = plot(
    title = "Posterior over effect of PDI",
    grid = false,
    showaxis = false,
    bottom_margin = -30Plots.px,
)
plot(
    title,
    density(chns[Symbol("learning_rate.β[2]")], title = "learning rate", label = nothing),
    density(chns[Symbol("action_noise.β[2]")], title = "action noise", label = nothing),
    layout = @layout([A{0.01h}; [B C]])
)

# We can also plot the posterior over the effect of session number on the action model parameters.
# Here, it looks like there is a negative effect of session number on the action noise, so that participants become more consistent in their responses over the course of the experiment.

title = plot(
    title = "Posterior over effect of session",
    grid = false,
    showaxis = false,
    bottom_margin = -30Plots.px,
)
plot(
    title,
    density(chns[Symbol("learning_rate.β[3]")], title = "learning rate", label = nothing),
    density(chns[Symbol("action_noise.β[3]")], title = "action noise", label = nothing),
    layout = @layout([A{0.01h}; [B C]])
)


# We can also extract the session parameters from the model.
session_parameters = get_session_parameters!(model);

#TODO: plot the session parameters

# And we can extract a dataframe with the median of the posterior for each session parameter
parameters_df = summarize(session_parameters, median)
show(parameters_df)

# We can also look at the implied state trajectories, in this case the expected value.
state_trajectories = get_state_trajectories!(model, :expected_value);

#TODO: plot the state trajectories

# These can also be summarized in a dataframe, for downstream analysis.
states_df = summarize(state_trajectories, median)
show(states_df)




# ## Comparing to a simple random model
# We can compare the Rescorla-Wagner model to a simple random model, which samples actions randomly from a Gaussian distribution with a fixed mean $\mu$ and standard deviation $\sigma$.


#First we create the simple model
function gaussian_random(attributes::ModelAttributes, observation::Float64)

    parameters = load_parameters(attributes)

    σ = parameters.std
    μ = parameters.mean

    return Normal(μ, σ)
end

action_model = ActionModel(
    gaussian_random,
    observations = (; observation = Observation()),
    actions = (; report = Action(Normal)),
    parameters = (std = Parameter(1), mean = Parameter(50)),
)

# ### Fitting the model
# We also set priors for this simpler model.
# Here we set the priors separately for the mean and the noise, since they are on very different scales.
# We center the priors for the mean at 50, as this is the middle of the range of actions.
# The priors for the noise are similar to those used with the Rescorla-Wagner model.

mean_regression_prior = RegressionPrior(
    β = [Normal(50, 10), Normal(0, 10), Normal(0, 10)],
    σ = truncated(Normal(0, 10), lower = 0),
)
noise_regression_prior = RegressionPrior(
    β = [Normal(0, 0.3), Normal(0, 0.2), Normal(0, 0.5)],
    σ = truncated(Normal(0, 0.3), lower = 0),
)

population_model = [
    Regression(@formula(mean ~ 1 + pdi_total + session + (1 | ID)), mean_regression_prior),
    Regression(
        @formula(std ~ 1 + pdi_total + session + (1 | ID)),
        exp,
        noise_regression_prior,
    ),
]

simple_model = create_model(
    action_model,
    population_model,
    JGET_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

## Set AD backend ##
using ADTypes: AutoForwardDiff
adtype = AutoForwardDiff();

## Fit model ##
chns = sample_posterior!(simple_model, MCMCThreads(), n_chains = 2, n_samples = 500, adtype = adtype)

#Plot the posteriors
plot(
    plot(
        plot(
            title = "Posterior over effect of PDI",
            grid = false,
            showaxis = false,
            bottom_margin = -30Plots.px,
        ),
        density(
            chns[Symbol("mean.β[2]")],
            title = "mean",
            label = nothing,
        ),
        density(
            chns[Symbol("std.β[2]")],
            title = "std",
            label = nothing,
        ),
        layout = @layout([A{0.01h}; [B C]])
    ),
    plot(
        plot(
            title = "Posterior over effect of session",
            grid = false,
            showaxis = false,
            bottom_margin = -30Plots.px,
        ),
        density(chns[Symbol("mean.β[3]")], title = "mean", label = nothing),
        density(chns[Symbol("std.β[3]")], title = "std", label = nothing),
        layout = @layout([A{0.01h}; [B C]])
    ),
    layout = (2,1)
)

# And we can use model comparison to compare the two models.
#TODO: model comparison