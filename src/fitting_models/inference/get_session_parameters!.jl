"""
    get_session_parameters!(modelfit::ModelFit, prior_or_posterior::Symbol = :posterior; verbose::Bool = true)

Extract posterior or prior samples of session-level parameters from a fitted model.

If the requested samples have not yet been drawn, this function will call `sample_posterior!` or `sample_prior!` as needed. Returns a `SessionParameters` struct containing the samples for each session and parameter.

# Arguments
- `modelfit::ModelFit`: The fitted model object.
- `prior_or_posterior::Symbol = :posterior`: Whether to extract from the posterior (`:posterior`) or prior (`:prior`).
- `verbose::Bool = true`: Whether to print warnings if sampling is triggered.

# Returns
- `SessionParameters`: Struct containing samples for each session and parameter.

# Example
```jldoctest; setup = :(using ActionModels, DataFrames, StatsPlots; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = (; learning_rate = LogitNormal()); model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id); chns = sample_posterior!(model, sampler = HMC(0.8, 10),n_samples=100, n_chains=1, progress = false))
julia> params = get_session_parameters!(model);

julia> params isa ActionModels.SessionParameters
true
```

# Notes
- Use `prior_or_posterior = :prior` to extract prior samples instead of posterior.
- The returned object can be summarized with `MCMCChains.summarize`.
"""
function get_session_parameters!(
    modelfit::ModelFit,
    prior_or_posterior::Symbol = :posterior;
    verbose::Bool = true,
)

    #Extract appropriate sample result
    if prior_or_posterior == :posterior

        if verbose && isnothing(modelfit.posterior)
            @warn "Posterior has not yet been sampled. Sampling now with default settings. Use sample_posterior! to choose other settings."
        end

        sample_posterior!(modelfit)

        sample_result = modelfit.posterior

    elseif prior_or_posterior == :prior

        if verbose && isnothing(modelfit.prior)
            @warn "sampling from the prior..."
        end

        sample_prior!(modelfit)

        sample_result = modelfit.prior

    else
        @error "use only either :posterior or :prior as the second argument"
    end

    #If the session_parameters have already been extracted, return them
    if !isnothing(sample_result.session_parameters)
        return sample_result.session_parameters
    end

    #Extract model & chains
    model = modelfit.model
    chains = sample_result.chains

    #Extract the session parameters (sample x chain array of vectors with parameter tuples for each session)
    returned_values = returned(model.args.population_model, chains)

    #Extract info
    estimated_parameter_names = modelfit.info.estimated_parameter_names
    action_model = model.args.action_model
    n_parameters = length(estimated_parameter_names)
    session_ids = modelfit.info.session_ids
    n_sessions = length(session_ids)
    n_samples, n_chains = size(returned_values)

    #Extract parameter types
    parameter_types =
        merge(get_parameter_types(action_model), get_parameter_types(action_model.submodel))

    #Create an empty AxisArray for each parameter
    session_parameters = NamedTuple(
        parameter_name => NamedTuple(
            Symbol(session_id) => AxisArray(
                Array{parameter_types[parameter_name]}(undef, n_samples, n_chains),
                Axis{:sample}(1:n_samples),
                Axis{:chain}(1:n_chains),
            ) for session_id in session_ids
        ) for parameter_name in estimated_parameter_names
    )

    #Go through each sampled parameter
    @progress for sample_idx = 1:n_samples
        for chain_idx = 1:n_chains

            sample_parameters = collect(returned_values[sample_idx, chain_idx])

            for session_idx = 1:n_sessions

                parameters = sample_parameters[session_idx]

                for parameter_idx = 1:n_parameters

                    parameter_value = parameters[parameter_idx]

                    #And set it in the appropriate place
                    session_parameters[parameter_idx][session_idx][sample_idx, chain_idx] =
                        parameter_value
                end
            end
        end
    end

    #Store as SessionParameters struct
    session_parameters = SessionParameters(
        session_parameters,
        modelfit,
        estimated_parameter_names,
        session_ids,
        parameter_types,
        n_samples,
        n_chains,
    )

    #Save the session parameters
    sample_result.session_parameters = session_parameters

    return session_parameters
end
