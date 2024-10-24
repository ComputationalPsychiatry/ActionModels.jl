
"""
Function for performing a single parameter recovery (simulation and fitting)
"""
function single_recovery(
    original_agent::Agent,
    sampler_settings::NamedTuple,
    parameters::Dict,
    prior_and_idx::Tuple,
    input_sequence_and_idx::Tuple,
    simulation_idx::Int;
    check_parameter_rejections::Bool = false,
)
    #Make a copy of the original agent to avoid changing it
    agent = deepcopy(original_agent)

    #Extract the prior and input sequence
    prior_idx, prior = prior_and_idx
    input_sequence_idx, input_sequence = input_sequence_and_idx

    #Make a renamed dictionary with true parameter values
    true_parameters = Dict()
    #For each parameter
    for key in keys(parameters)
        #Concatenate tuples
        if key isa Tuple
            new_key = join(key, tuple_separator)
        else
            new_key = key
        end
        #Add prefix to show that it is an estimate
        new_key = "generative" * id_separator * new_key
        #Save in the new dict
        true_parameters[new_key] = parameters[key]
    end

    #Set the parameters for the agent
    set_parameters!(agent, parameters)
    reset!(agent)
    
    #In the normal case
    if !check_parameter_rejections
        #Give inputs and get simulated actions
        simulated_actions = give_inputs!(agent, input_sequence)

    #If there is a check for parameter rejections
    else
        #Look for errors
        try
            #Give inputs and get simulated actions
            simulated_actions = give_inputs!(agent, input_sequence)
        #If there is an error
        catch err
            #If it is a parameter rejection
            if err isa RejectParameters

                #Return a dataframe without any parameter estimates
                return hcat(
                    DataFrame(true_parameters),
                    DataFrame(
                    prior_idx = prior_idx,
                    input_sequence_idx = input_sequence_idx,
                    simulation_idx = simulation_idx,)
                    )

            else
                rethrow(err)
            end
        end
    end

    #Create model
    model = create_model(agent, prior, input_sequence, simulated_actions; check_parameter_rejections = check_parameter_rejections)

    #Fit the model to the simulated data
    result = fit_model(model; sampler_settings..., progress = false)

    #Extract the agent posteriors
    agent_parameters = extract_quantities(model, result.chains)
    posterior_medians = first(values(get_estimates(agent_parameters, Dict)))

    ## - Rename dictionaries - ##
    #Make a renamed dictionary with estimated parameter values
    estimated_parameters = Dict()
    #For each parameter
    for key in keys(posterior_medians)
        #Transform the tuple key to a string
        new_key = string(key)
        #Add prefix to show that it is an estimate
        new_key = "estimated" * id_separator * new_key
        #Save in the new dict
        estimated_parameters[new_key] = posterior_medians[key]
    end

    #Gather into a dataframe row
    dataframe_row = hcat(
        DataFrame(true_parameters),
        DataFrame(estimated_parameters),
        DataFrame(
            prior_idx = prior_idx,
            input_sequence_idx = input_sequence_idx,
            simulation_idx = simulation_idx,
        ),
    )
end


"""
Function for performing parameter recovery
"""
function parameter_recovery(
    agent::Agent,
    parameter_ranges::Dict,
    input_sequences::Array{V},
    priors::Union{P,Vector{P}},
    n_simulations::Int;
    sampler_settings::NamedTuple = (),
    parallel::Bool = false,
    show_progress::Bool = true,
    check_parameter_rejections::Bool = false,
) where {K<:Any,D<:Distribution,P<:Dict{K,D},V<:Any}

    ## - Format input - ##
    # If priors is a single dictionary, convert it to a vector of dictionaries
    if !(priors isa Vector)
        priors = [priors]
    end

    # If input_sequences is a single vector, convert it to a vector of vectors
    if !(V <: Array)
        input_sequences = [input_sequences]
    end

    ## - prepare parameter combinations - ##
    # Create two arrays, param_keys and param_values, to store the keys and values of the parameter_ranges dictionary
    param_keys = collect(keys(parameter_ranges))
    param_values = collect(values(parameter_ranges))

    # Generate all possible combinations of parameter values using the Cartesian product of param_values
    combinations = collect(Iterators.product(param_values...))

    # Create an empty vector called parameter_combinations to store dictionaries of parameter combinations
    parameter_combinations = Vector{Dict}()

    # Iterate over each combination of parameter values
    for combination in combinations
        # Create a dictionary called dict_from_vectors by pairing each key in param_keys with its corresponding value in combination
        dict_from_vectors =
            Dict(param_keys[i] => combination[i] for i = 1:length(param_keys))

        # Add the dictionary dict_from_vectors to the parameter_combinations vector
        push!(parameter_combinations, dict_from_vectors)
    end


    ## SIMULATION ##
    #Construct all combinations of parameters, priors, input sequences, one for each simulation
    recovery_infos = collect(
        Iterators.product(
            parameter_combinations,
            enumerate(priors),
            enumerate(input_sequences),
            1:n_simulations,
        ),
    )

    #If parallelization is used
    if parallel
        #Use pmap
        map_function = pmap
    else
        #Use map
        map_function = map
    end

    #Toggle whether to show progress bar
    if show_progress
        #Run simulations and parameter fits in parallel
        outcome = @showprogress map_function(
            recovery_info -> single_recovery(agent, sampler_settings, recovery_info...; check_parameter_rejections = check_parameter_rejections),
            recovery_infos,
        )
    else
        #Run simulations and parameter fits in parallel
        outcome = map_function(
            recovery_info -> single_recovery(agent, sampler_settings, recovery_info...; check_parameter_rejections = check_parameter_rejections),
            recovery_infos,
        )
    end

    # Concatenate the outcomes into a single dataframe
    outcome = vcat(outcome..., cols = :union)

    return outcome
end
