# # Creating a custom fit_model() function

# If you wish to alter the fit_model() function to you own specific use case we have provided a compact version of the function with the basic structure for you to work on.



#  ## Strucure of create\_agent\_model is the following:


# # we start by specifying function name and the inputs
# @model function full_model(
#     agent,
#     param_priors,
#     inputs,
#     actions,
#     impute_missing_actions,
# )

#     ##Initialize dictionary for storing sampled parameters
#     fitted_parameters = Dict()

#     ##--------- Sample parameters from the priors set, and set the parameters in the agent ---------

#     ##Give Turing prior distributions for each fitted parameter
#     for (param_key, param_prior) in param_priors
#         fitted_parameters[param_key] ~ param_prior
#     end

#     ##Set agent parameters to the sampled values
#     set_parameters!(agent, fitted_parameters)
#     reset!(agent)

#     ## ----------- Specify settings for whether inputs are as vectors or arrays ---------

#     ##If the input is a single vector
#     if inputs isa Vector
#         ##Prepare to through one value at a time
#         iterator = enumerate(inputs)
#     else
#         ##For an array, go through each row
#         iterator = enumerate(eachrow(inputs))
#     end


#     ## ---------- Go through inputs and get the action probability distribution from the agent ------

#     ##For each timestep and input
#     for (timestep, input) in iterator
#         ##If no errors occur
#         try

#             ##Get the action probability distribution from the action model
#             action_probability_distribution = agent.action_model(agent, input)

#             ## ---- if one single action is made at each timestep (if only one action model is specified in the configurations ) ----

#             if actions isa Vector

#                 ##If the action isn't missing, or if missing actions are to be imputed
#                 if !ismissing(actions[timestep]) || impute_missing_actions
#                     #Pass it to Turing
#                     actions[timestep] ~ action_probability_distribution
#                 end

#                 ## ---- if multiple actions are made at each timestep (if more action models are specified in the configurations ) ----

#             elseif actions isa Array

#                 ##Go throgh each action distribution
#                 for (action_indx, distribution) in
#                     enumerate(action_probability_distribution)

#                     ##If the action isn't missing, or if missing actions are to be imputed
#                     if !ismissing(actions[timestep, action_indx]) || impute_missing_actions
#                         ##Pass it to Turing
#                         actions[timestep, action_indx] ~ distribution
#                     end
#                 end
#             end

#             ##--------- If an error occurs -----------

#         catch e
#             ##If the custom errortype RejectParameters occurs
#             if e isa RejectParameters
#                 ##Make Turing reject the sample
#                 Turing.@addlogprob!(-Inf)
#             else
#                 ##Otherwise, just throw the error
#                 throw(e)
#             end
#         end
#     end
# end


# #  ## Strucure of fit_model()

# # The different elements in the code are seperated.

# # start by inititializing function name an inputs
# function fit_model(
#     agent::Agent,
#     param_priors::Dict,
#     inputs::Array,
#     actions::Array;
#     fixed_parameters::Dict = Dict(),
#     sampler = NUTS(),
#     n_cores::Integer = 1,
#     n_iterations::Integer = 1000,
#     n_chains = 2,
#     verbose = true,
#     show_sample_rejections = false,
#     impute_missing_actions::Bool = false,
# )
#     ##Store old parameters for resetting the agent later
#     old_parameters = get_parameters(agent)


#     ## -------------- CHECKS START ---------------- #

#     ##Unless warnings are hidden
#     if verbose
#         ##If there are any of the agent's parameters which have not been set in the fixed or sampled parameters
#         if any(
#             key -> !(key in keys(param_priors)) && !(key in keys(fixed_parameters)),
#             keys(old_parameters),
#         )
#             @warn "the agent has parameters which are not specified in the fixed or sampled parameters. The agent's current parameter values are used as fixed parameters"
#         end

#         ##If a parameter has been specified both in the fixed and sampled parameters
#         if any(key -> key in keys(fixed_parameters), keys(param_priors))
#             @warn "one or more parameters have been specified both in the fixed and sampled parameters. The fixed parameter value is used"

#             ##Remove the parameter from the sampled parameters
#             for key in keys(fixed_parameters)
#                 if key in keys(param_priors)
#                     delete!(param_priors, key)
#                 end
#             end
#         end
#     end

#     ##If there are no parameters to sample
#     if length(param_priors) == 0
#         ##Throw an error
#         throw(
#             ArgumentError(
#                 "no parameters to sample. Either an empty dictionary of parameter priors was passed, or all parameters with priors were also specified as fixed parameters",
#             ),
#         )
#     end

#     ##If there are different amounts of inputs and actions
#     if size(inputs, 1) != size(actions, 1)
#         throw(
#             ArgumentError(
#                 "there are different amounts of inputs and actions (they differ in their first dimension). This is not supported",
#             ),
#         )
#     end

#     ### Set fixed parameters to agent ###
#     set_parameters!(agent, fixed_parameters)

#     ### Run forward once ###
#     ##Initialize dictionary for populating with median parameter values
#     sampled_parameters = Dict()
#     ##Go through each of the agent's parameters
#     for (param_key, param_prior) in param_priors
#         ##Add the median value to the tuple
#         sampled_parameters[param_key] = median(param_prior)
#     end
#     ##Set sampled parameters
#     set_parameters!(agent, sampled_parameters)
#     ##Reset the agent
#     reset!(agent)

#     try
#         ##Run it forwards
#         test_actions = give_inputs!(agent, inputs)

#         ##If the model returns a different amount of actions from what was inputted
#         if size(test_actions) != size(actions)
#             throw(
#                 ArgumentError(
#                     "the passed actions is a different shape from what the model returns",
#                 ),
#             )
#         end

#     catch e
#         ##If a RejectParameters error occurs
#         if e isa RejectParameters
#             ##Warn the user that prior median parameter values gives a sample rejection
#             if verbose
#                 @warn "simulating with median parameter values from the prior results in a rejected sample."
#             end
#         else
#             ##Otherwise throw the actual error
#             throw(e)
#         end
#     end


#     ##--------------- FIT MODEL START ----------- #

#     ##------- Fit model with one core specified ------ #

#     ##If only one core is specified, use sequentiel sampling
#     if n_cores == 1

#         ##Initialize Turing model
#         model =
#             full_model(agent, param_priors, actions, inputs, impute_missing_actions)

#         ##If sample rejection warnings are to be shown
#         if show_sample_rejections
#             ##Fit model to inputs and actions, as many separate chains as specified
#             chains = map(i -> sample(model, sampler, n_iterations), 1:n_chains)

#             ##If sample rejection warnings are not to be shown
#         else
#             ##Create a logger which ignores messages below error level
#             sampling_logger = Logging.SimpleLogger(Logging.Error)
#             ##Use that logger
#             chains = Logging.with_logger(sampling_logger) do

#                 ##Fit model to inputs and actions, as many separate chains as specified
#                 map(i -> sample(model, sampler, n_iterations), 1:n_chains)
#             end
#         end

#         ##------- Fit model with one multiple cores specified ------ #

#     elseif n_cores > 1

#         ##If the user has already created processsses
#         if length(procs()) == 1
#             ##Add worker processes
#             addprocs(n_cores, exeflags = "--project")

#             ##Set flag to remove the workers later
#             remove_workers_at_end = true
#         else
#             ##Error
#             @warn """
#             n_cores was set to > 1, but workers have already been created. No new workers were created, and the existing ones are used for parallelization.
#             Note that the following variable names are broadcast to the workers: sampler agent param_priors inputs actions impute_missing_actions
#             """
#             #Set flag to not remove the workers later
#             remove_workers_at_end = false
#         end

#         ##------ Setup distribution of information to processes for parallellization -----

#         ##Load packages on worker processes
#         @everywhere @eval using ActionModels
#         @everywhere @eval using Turing

#         ##Broadcast necessary information to workers
#         @everywhere sampler = $sampler
#         @everywhere agent = $agent
#         @everywhere param_priors = $param_priors
#         @everywhere inputs = $inputs
#         @everywhere actions = $actions
#         @everywhere impute_missing_actions = $impute_missing_actions

#         ##----- fit model to inputs with shown sample rejection warnings ------

#         if show_sample_rejections
#             ##Fit model to inputs and actions, as many separate chains as specified
#             chains = pmap(
#                 i -> sample(
#                     full_model(
#                         agent,
#                         param_priors,
#                         inputs,
#                         actions,
#                         impute_missing_actions,
#                     ),
#                     sampler,
#                     n_iterations,
#                     save_state = false,
#                 ),
#                 1:n_chains,
#             )

#             ##----- fit model to inputs not showing sample rejection warnings ------

#         else
#             ##Create a logger which ignores messages below error level
#             sampling_logger = Logging.SimpleLogger(Logging.Error)
#             ##Use that logger
#             chains = Logging.with_logger(sampling_logger) do

#                 ##Fit model to inputs and actions, as many separate chains as specified
#                 pmap(
#                     i -> sample(
#                         full_model(
#                             agent,
#                             param_priors,
#                             inputs,
#                             actions,
#                             impute_missing_actions,
#                         ),
#                         sampler,
#                         n_iterations,
#                         save_state = false,
#                     ),
#                     1:n_chains,
#                 )
#             end
#         end

#         ## ----- end configurations: worker cleanup and combining chains -------
#         ##If workers are to be removed
#         if remove_workers_at_end
#             ##Remove workers
#             rmprocs(workers())
#         end

#     else
#         throw(
#             ArgumentError(
#                 "n_cores was set to a non-positive integer. This is not supported.",
#             ),
#         )
#     end

#     ##Concatenate chains together
#     chains = chainscat(chains...)

#     ##-------------- CLEANUP ----------------

#     ##Reset the agent to its original parameters
#     set_parameters!(agent, old_parameters)
#     reset!(agent)

#     ##Turing includes the dictionary name 'fitted_parameters' in the parameter names, so it must be removed
#     ##Initialize dict for replacement names
#     replacement_param_names = Dict()
#     ##For each parameter
#     for param_key in keys(param_priors)
#         ##Set to replace the fitted_parameters[] version with just the parameter name
#         replacement_param_names["fitted_parameters[$param_key]"] = param_key
#     end
#     ##Input the dictionary to replace the names
#     chains = replacenames(chains, replacement_param_names)

#     return chains
# end
