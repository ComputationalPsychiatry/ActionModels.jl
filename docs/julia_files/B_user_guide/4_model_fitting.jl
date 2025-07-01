# # Fitting action models to data

# In this section, we will cover how to fit action models to data with ActionModels.
# First, we will demonstrate how to set up the model that will be fit under hood by relying on Turing.
# This consists of specifying three things: an action model, a population model, and the data to fit the model to.
# Then, we will show how to sample from the posterior and prior distribution of the model parameters.
# And finally, we will show the tools in ActionModels to inspect and extract the results of the model fitting.

# ## Setting up the model
# First we import ActionModels, as well as StatsPlots for plotting the results later.
using ActionModels, StatsPlots

# ### Defining the action model
# We will here use the premade Rescorla-Wagner action model provided by ActionModels. This is identical to the model described in the defining action models REF section.
action_model = ActionModel(RescorlaWagner())

# ### Loading the data
# We will then specify the data that we want to fit the model to.
# For this example, we will use a simple manually created dataset, where three participants have completed an experiment where they must predict the next location of a moving target.
# Each participant has completed the experiment twice, in a control condition and under and experimental treatment.
using DataFrames

data = DataFrame(
    observations = repeat([1.0, 1, 1, 2, 2, 2], 6),
    actions = vcat(
        [0, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0, 0.5, 0.8, 1, 1.5, 1.8],
        [0, 2, 0.5, 4, 5, 3],
        [0, 0.1, 0.15, 0.2, 0.25, 0.3],
        [0, 0.2, 0.4, 0.7, 1.0, 1.1],
        [0, 2, 0.5, 4, 5, 3],
    ),
    id = vcat(
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
        repeat(["A"], 6),
        repeat(["B"], 6),
        repeat(["C"], 6),
    ),
    treatment = vcat(repeat(["control"], 18), repeat(["treatment"], 18)),
)

show(data)

# ### Specifying the population model
# Finally, we will specify a population model, which is the model of how parameters vary between the different sessions in the data.
# There are various options when doing this, which are described in the population model REF section.
# Here, we will use a regression population model, where we assume that the learning rate and action noise parameters depend linearly on the experimental treatment.
# It is a hierarchical model, which means assuming that the parameters are sampled from a Gaussian distribution, where the mean of the distribution is a linear function of the treatment condition.
# This is specified with standard LMER syntax, and we use a logistic inverse link function for the learning rate to ensure that it is between 0 and 1, and an exponential inverse link function for the action noise to ensure that it is positive.
population_model = [
    Regression(@formula(learning_rate ~ treatment + (1 | id)), logistic),
    Regression(@formula(action_noise ~ treatment + (1 | id)), exp),
];

# ### Creating the full model
# Finally, we can combine the three components into a full model that can be fit to the data.
# This is done with the `create_model` function, which takes the action model, population model, and data as arguments.
# Additionally, we specify which columns in the data contain the actions, observations, and session identifiers.
# This creates an ActionModels.ModelFit object, which containts the full model, and will contain the sampling results after fitting.
model = create_model(
    action_model,
    population_model,
    data;
    action_cols = :actions,
    observation_cols = :observations,
    session_cols = [:id, :treatment],
)

# If there are multiple actions or observations, we can specify them as a NamedTuple mapping each action or observation to a column in the data.
# For example, if we had two actions and two observations, we could specify them as follows:

(action_name1 = :action_column_name1, action_name2 = :action_column_name2);

# The column names can also be specified as a vector of symbols, in which case it will be assumed that the order matches the order of actions or observations in the action model.

# Finally, there may be missing data in the dataset.
# If actions are missing, they can be imputed by ActionModels. This is done by setting the `impute_actions` argument to `true` in the `create_model` function.
# If `impute_actions` is not set to `true`, missing actions will simply be skipped during sampling instead.
# This is a problem for action models which depend on their previous actions.

# ## Fitting the model

# Now that the model is created, we are ready to fit it.
# This is done under the hood using MCMC sampling, which is provided by the Turing.jl framework.
# ActionModels provides the `sample_posterior!` function, which fits the model in this way with sensible defaults.

chns = sample_posterior!(model)

# This returns a MCMCChains Chains object, which contains the samples from the posterior distribution of the model parameters.
# For each parameter, there is a posterior for the β values (intercept and treatment effect), as well as for the deviation of the random effects σ and the single random effects $r$.
# Notably, with a fulle dataset, the posterior will contain a large number of parameters. 
# We can see that the second beta value for the learning rate (the dependence on the treatment condition) is negative. The dataset has been constructed to have lower learning rates in the treatment condition, so this is expected.

# Notably, `sample_posterior!` has many options for how to sample the posterior, which can be set with keyword arguments.
# If we pass either `MCMCThreads()` or `MCMCDistributed()` as the second argument, Turing will use multithreading or distributed sampling to parallellise between chains.
# It is recommended to use MCMCThreads for multithreading, but note that Julia must be started with the `--threads` flag to enable multithreading.
# We can specify the number of samples and chains to sample with the `n_samples` and `n_chains` keyword arguments.
# The `init_params` keyword argument can be used to specify how the initial parameters for the chains are set.
# It can be set to `:MAP` or `:MLE` to use the maximum a posteriori or maximum likelihood estimates as the initial parameters, respectively.
# It can be set to `:sample_prior` to draw a single sample from the prior distribution, or to `nothing` to use Turing's default of random values between -2 and 2 as the initial parameters.`
# Finally, a vector of initial parameters can be passed, which will be used as the initial parameters for all chains.
# Other arguments for the sampling can also be passed. This includes the autodifferentiation backend to use, which can be set with the `ad_type` keyword argument, and the sampler to use, which can be set with the `sampler` keyword argument.
# Notably, `sample_posterior!` will return the already sampled `Chains` object if the posterior has already been sampled. Set `resample = true` to override the already sampled posterior.

chns = sample_posterior!(
    model,
    MCMCThreads(),
    n_samples = 500,
    n_chains = 4,
    init_params = :MAP,
    ad_type = AutoForwardDiff(),
    sampler = NUTS(),
    resample = true,
);

# Finally, some users may wish to use Turing's own interface for sampling from the posterior instead.
# The Turing inferface is more flexible in general, but requires more boilerplate code to set up.
# For this case, the `ActionModels.ModelFit` objects contains the Turing model that is used under the hood. Users can extract and use it as any other Turing model, if they wish.

turing_model = model.model

# If users want to sample from the model themselves, but still want to draw on the rest of the ActionModels API, they can set it in the ModelFit object themselves by creating an `ActionModels.ModelFitResult` object.
# This should be passed to either the posterior or prior field of the `ModelFit` object, after which it will interface with the ActionModels API as normal.
using ActionModels: Turing
chns = sample(turing_model, NUTS(), 1000, progress = false);

model.posterior = ActionModels.ModelFitResult(; chains = chns);


# In addition to sampling from the posterior, ActionModels also provides functionality for sampling from the prior distribution of the model parameters.
# This is done with the `sample_prior!` function, which works in a similar way to `sample_posterior!`.
# Notably, it is much simpler, due to not requiring a complex sampler. This means that it only takes `n_chains` and `n_samples` as keyword arguments.

prior_chns = sample_prior!(model, n_chains = 1, n_samples = 1000)

# ## Investigating the results
# ### Population model parameters
# The first step in investigating model fitting results is often to look at the population model parameters.
# Population model parameters and how to visualize them will depend on the type of population model used.
# See the population model REF section for more details on how to interpret results from different population models.
# But in general, the Chains object returned by `sample_posterior!` will contain the posterior distribution of the population model parameters.
# These can be visualized with various plotting functions; see the [MCMCChains documentation](https://turinglang.org/MCMCChains.jl/stable/statsplots/) for an overview.
# Here, we just use the standard plot function to visualize the posterior distribution over the beta values of interest:

#Fit the model
chns = sample_posterior!(model);

# Plot the posterior distribution of the learning rate and action noise beta parameters.
plot(chns[[Symbol("learning_rate.β[1]"), Symbol("learning_rate.β[2]"), Symbol("action_noise.β[1]"), Symbol("action_noise.β[2]")]])

#TODO: ArviZ support / specifalized plotting functions

# ### Parameter per session
# Beyond the population model parameters, users will often be intersted in the parameter estimates for each session in the data.
# The session parameters can be extracted with the `get_session_parameters!` function, which returns a `ActionModels.SessionParameters` object.
# Whether session parameter estimates should be extracted for the posterior or prior distribution can be specified as the second argument.

#Extract posterior and prior estimates for session paramaters
session_parameters = get_session_parameters!(model)
prior_session_parameters = get_session_parameters!(model, :prior)

# ActioModels provides a convenient functionality for plotting the session parameters.
#TODO: plot(session_parameters)

# Users can also access the full distribution over the session parameters, for use in manual downstream analysis.
# The `ActionModels.SessionParameters` object contains the distributions for each parameter and each session. 
# These can be found in the value field, which contains nested `NamedTuples` for each parameter and session, and ultimately an `AxisArray` with the samples for each chain.

learning_rate_singlesession = getfield(session_parameters.value.learning_rate, Symbol("id:A.treatment:control"));

# The user may visualize, summarize or analyze the samples in whichever way they prefer.

#Calcualate the mean
mean(learning_rate_singlesession)
#Plot the distribution
density(learning_rate_singlesession, title = "Learning rate for session A in control condition")

# ActionModels provides a convenient function for summarizing all the session parameters as a `DataFrame`, that can be used for further analysis.

median_df = summarize(session_parameters)

show(median_df)

# This returns the median of each parameter for each session. 
# The user can pass other functions for summarizing the samples, as for example to calculate the standard deviation of the posterior.

std_df = summarize(session_parameters, std)

show(std_df)

# This can be saved to disk or used for plotting or analysis in whichever way the user prefers.

# ### State trajectories per session
# Users can also extract the estimated trajectory of states for each session with the `get_state_trajectories!` function, which returns an `ActionModels.StateTrajectories` object.
# State trajectories are often used to correlate with some external measure, such as neuroimaging data.
# The second argument specifies which state to extract. Again, the user can also specify to have the prior state trajectories extracted by passing `:prior` as the second argument.
state_trajectories = get_state_trajectories!(model, :expected_value)

prior_state_trajectories = get_state_trajectories!(model, :expected_value, :prior)

# ActionModels also provides functionality for plotting the state trajectories.
#TODO: plot(state_trajectories)

# The `ActionModels.StateTrajectories` object contains the prior or porsterior distribution over state trajectories for each session, and for each of the specified states.
# This is stored in the value field, which contains nested `NamedTuple` objects with state names and then session names as keys, and ultimately an AxisArray with the samples across timesteps.
# This AxisArray has three axes: the first is the sample index, the second is the session index, and the third is the timestep index.

expected_value_singlesession = getfield(state_trajectories.value.expected_value, Symbol("id:A.treatment:control"));

# Again, we can visualize, summarize or analyze the samples in whichever way we prefer.
# Here, we chose the second timestep of the first session, and calculate the mean and plot the distribution.

mean(expected_value_singlesession[timestep=2]) 
density(expected_value_singlesession[timestep=2], title = "Expectation at time 2 session A control condition")

# This can also be summarized neatly in a DataFrame:

median_df = summarize(state_trajectories, median)

show(median_df)
#And from here, it can be used for plotting or further analysis as desired by the user.
