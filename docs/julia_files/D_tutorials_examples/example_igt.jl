# # Iowa Gambling Task
# In this tutorial, we will fit the PVL-Delta model to data from the Iowa Gambling Task (IGT) using the ActionModels package.
# In the IGT, participants choose cards from four decks, each with different reward and loss probabilities, and must learn over time which decks are advantageous.
# We will use data from Ahn et al. (2014), which includes healthy controls and participants with heroin or amphetamine addictions.
# There are more details about the collected data in the `docs/example_data/ahn_et_al_2014/ReadMe.txt`

# ## Loading data
# First, we load the ActionModels package. We also load CSV and Dataframes for loading the data, and StatsPlots for plotting the results.
using ActionModels
using CSV, DataFrames
using StatsPlots

# Then we load the ahn et al. (2014) data, which is available in the `docs/example_data/ahn_et_al_2014` folder.

ActionModels_path = dirname(dirname(pathof(ActionModels))) #hide
docs_path = joinpath(ActionModels_path, "docs") #hide
#Import data
data_healthy = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_healthy_control.txt"),
    DataFrame,
)
data_healthy[!, :clinical_group] .= "1_control"
data_heroin = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_heroin.txt"),
    DataFrame,
)
data_heroin[!, :clinical_group] .= "2_heroin"
data_amphetamine = CSV.read(
    joinpath(docs_path, "example_data", "ahn_et_al_2014", "IGTdata_amphetamine.txt"),
    DataFrame,
)
data_amphetamine[!, :clinical_group] .= "3_amphetamine"

#Combine into one dataframe
ahn_data = vcat(data_healthy, data_heroin, data_amphetamine)
ahn_data[!, :subjID] = string.(ahn_data[!, :subjID])

#Make column with total reward
ahn_data[!, :reward] = Float64.(ahn_data[!, :gain] + ahn_data[!, :loss]);

show(ahn_data)

# For this example, we will subset the data to only include two subjects from each clinical group.
# This makes the runtime much shorter. Simply skip this step if you want to use the full dataset.

ahn_data =
    filter(row -> row[:subjID] in ["103", "117", "105", "136", "130", "149"], ahn_data);

# ## Creating the model
# Then we construct the model to be fitted to the data. 
# We use the PVL-Delta action model, which is a classic model for the IGT.
# The PVL-Delta is a type of reinfrocement learning model that learns the expected value for each of the decks in the IGT.
# First, the observed reward is transformed with a prospect theory-based utlity curve.
# This means that the subjective value of a reward increses sub-linearly with reward magnitute, and that losses are weighted more heavily than gains.
# The expected value of each deck is then updated using a Rescorla-Wagner-like update rule.
# Finally, the probability of selecting each deck is calculated using a softmax function over the expected values of the decks, scaled by an inverse action noise parameter.
# In summary, the PVL-Delta has four parameters: the learning rate $\alpha$, the reward sensitivity $A$, the loss aversion $w$, and the action noise $\beta$.
# See the [section on the PVL-Delta premade model](./example_igt.md) in the documentation for more details.

# We create the PVL-Delta using the premade model from ActionModels.
# We specify the number of decks, and also that actions are selected before the expected values are updated.
# This is because in the IGT, at least as structured in this dataset, participants select a deck before they receive the reward and update expectations.
action_model = ActionModel(PVLDelta(n_options = 4, act_before_update = true))

# We then specify which column in the data corresponds to the action (deck choice) and which columns correspond to the observations (deck and reward).
# We also specify the columns that uniquely identify each session.
action_cols = :deck
observation_cols = (chosen_option = :deck, reward = :reward)
session_cols = :subjID;

# Finally, we create the full model. We use a hierarchical regression model to predict the parameters of the PVL-Delta model based on the clinical group (healthy, heroin, or amphetamine).
# First, we will set appropriate priors for the regression coefficients.
# For the action noise and the loss aversion, the outcome of the regression will be exponentiated before it is used in the model, so pre-transformed outcomes around 2 (exp(2) ≈ 7) are among the most extreme values to be expected.
# For the learning rate and reward sensitivity, we will use a logistic transformation, so pre-transformed outcomes around around 5 (logistic(5) ≈ 0.993) are among the most extreme values to be expected.

regression_prior = RegressionPrior(
    β = [Normal(0, 0.2), Normal(0, 0.3), Normal(0,0.3)],
    σ = truncated(Normal(0, 0.3), lower = 0),
)

plot(regression_prior.β[1], label = "Intercept")
plot!(regression_prior.β[2], label = "Effect of clinical group")
plot!(regression_prior.σ, label = "Random intercept std")
title!("Regression priors for the Rescorla-Wagner model")
xlabel!("Regression coefficient")
ylabel!("Density")


# We can now create the popluation model, and finally create the full model object.

population_model = [
    Regression(
        @formula(learning_rate ~ clinical_group + (1 | subjID)),
        logistic,
        regression_prior,
    ),
    Regression(
        @formula(reward_sensitivity ~ clinical_group + (1 | subjID)),
        logistic,
        regression_prior,
    ),
    Regression(
        @formula(loss_aversion ~ clinical_group + (1 | subjID)),
        exp,
        regression_prior,
    ),
    Regression(
        @formula(action_noise ~ clinical_group + (1 | subjID)),
        exp,
        regression_prior,
    ),
]

model = create_model(
    action_model,
    population_model,
    ahn_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

# ## Fitting the model
# We are now ready to fit the model to the data.
# For this model, we will use the Enzyme automatic differentiation backend, which is a high-performance automatic differentiation library. 
# Additionally, to keep the runtime of this tutorial short, we will only fit two chains with 500 samples.
# We will pass `MCMCThreas()` in order to parallelize the sampling across the two chains.
# This should take up to 10-15 minutes on a standard laptop. Switch to only a single thread for a learer progress bar.

#Set AD backend
using ADTypes: AutoEnzyme
import Enzyme: set_runtime_activity, Reverse
ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true));

#Fit model
chns = sample_posterior!(model, MCMCThreads(), init_params = :MAP, n_chains = 2, n_samples = 500, ad_type = ad_type)

# We can now inspect the results of the fitting process.
# We can plot the posterior distribution over the beta parameters of the regression model.
# We can see indications of lower reward sensitivity and lower loss aversion, as well as higher action noise, in the heroin and amphetamine groups compared to the healthy controls.
# Note that the posteriors would be more if we ha used the full dataset. 
plot(
    plot(
        plot(
            title = "Learning rate",
            grid = false,
            showaxis = false,
            bottom_margin = -30Plots.px,
        ),
        density(
            chns[Symbol("learning_rate.β[2]")],
            title = "Heroin",
            label = nothing,
        ),
        density(
            chns[Symbol("learning_rate.β[3]")],
            title = "Amphetamine",
            label = nothing,
        ),
        layout = @layout([A{0.01h}; [B C]])
    ),
    plot(
        plot(
            title = "Reward sensitivity",
            grid = false,
            showaxis = false,
            bottom_margin = -50Plots.px,
        ),
        density(
            chns[Symbol("reward_sensitivity.β[2]")],
            label = nothing,
        ),
        density(
            chns[Symbol("reward_sensitivity.β[3]")],
            label = nothing,
        ),
        layout = @layout([A{0.01h}; [B C]])
    ),
    plot(
        plot(
            title = "Loss aversion",
            grid = false,
            showaxis = false,
            bottom_margin = -50Plots.px,
        ),
        density(
            chns[Symbol("loss_aversion.β[2]")],
            label = nothing,
        ),
        density(
            chns[Symbol("loss_aversion.β[3]")],
            label = nothing,
        ),
        layout = @layout([A{0.01h}; [B C]])
    ),
    plot(
        plot(
            title = "Action noise",
            grid = false,
            showaxis = false,
            bottom_margin = -50Plots.px,
        ),
        density(
            chns[Symbol("action_noise.β[2]")],
            label = nothing,
        ),
        density(
            chns[Symbol("action_noise.β[3]")],
            label = nothing,
        ),
        layout = @layout([A{0.01h}; [B C]])
    ),
    layout = (4,1), size = (800, 1000),
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
# We can also compare the PVL-Delta to a simple random model, which randomly samples actions from a fixed Categorical distribution.

function categorical_random(
    attributes::ModelAttributes,
    chosen_option::Int64,
    reward::Float64,
)

    action_probabilities = load_parameters(attributes).action_probabilities

    return Categorical(action_probabilities)
end

action_model = ActionModel(
    categorical_random,
    observations = (chosen_option = Observation(Int64), reward = Observation()),
    actions = (; deck = Action(Categorical)),
    parameters = (; action_probabilities = Parameter([0.3, 0.3, 0.3, 0.1])),
)

# ### Fitting the model
# For this model, we use an independent session population model.
# We set the prior for the action probabilities to be a Dirichlet distribution, which is a common prior for categorical distributions.

population_model = (; action_probabilities = Dirichlet([1, 1, 1, 1]),)

simple_model = create_model(
    action_model,
    population_model,
    ahn_data,
    action_cols = action_cols,
    observation_cols = observation_cols,
    session_cols = session_cols,
)

#Set AD backend
using ADTypes: AutoEnzyme
import Enzyme: set_runtime_activity, Reverse
ad_type = AutoEnzyme(; mode = set_runtime_activity(Reverse, true));

#Fit model
chns = sample_posterior!(simple_model, n_chains = 1, n_samples = 500, ad_type = ad_type)

#TODO: plot the results

# We can also compare how well the PVL-Delta model fits the data compared to the simple random model.
#TODO: model comparison
