#########################################################
### SIMPLE POPULATION MODEL WITH INDEPENDENT SESSIONS ###
#########################################################
"""
    create_model(action_model::ActionModel, prior::NamedTuple, data::DataFrame; observation_cols, action_cols, session_cols=Vector{Symbol}(), population_model_type=IndependentPopulationModel(), verbose=true, kwargs...)

Create a hierarchical Turing model with independent session-level parameters for each session.

This function builds a model where each session's parameters are sampled independently from the specified prior distributions. Returns a `ModelFit` object ready for sampling and inference.

# Arguments
- `action_model::ActionModel`: The agent/action model to fit.
- `prior::NamedTuple`: Named tuple of prior distributions for each parameter (e.g., `(; learning_rate = LogitNormal())`).
- `data::DataFrame`: The dataset containing observations, actions, and session/grouping columns.
- `observation_cols`: Columns in `data` for observations. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `action_cols`: Columns in `data` for actions. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `session_cols`: Columns in `data` identifying sessions/groups (default: empty vector).
- `population_model_type`: Type of population model (default: `IndependentPopulationModel()`).
- `verbose`: Whether to print warnings and info (default: `true`).
- `kwargs...`: Additional keyword arguments passed to the underlying model constructor.

# Returns
- `ModelFit`: Struct containing the model, data, and metadata for fitting and inference.

# Example
```jldoctest; setup = :(using ActionModels, DataFrames; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); prior = (; learning_rate = LogitNormal()))
julia> model = create_model(action_model, prior, data; action_cols = :action, observation_cols = :observation, session_cols = :id);

julia> model isa ActionModels.ModelFit
true
```

# Notes
- Each session's parameters are sampled independently from the specified priors.
- The returned `ModelFit` object can be used with `sample_posterior!`, `sample_prior!`, and other inference utilities.
- Use this model when you do not want to share information across sessions/groups.
"""
function create_model(
    action_model::ActionModel,
    prior::NamedTuple{prior_names,<:Tuple{Vararg{Distribution}}},
    data::DataFrame;
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    session_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    population_model_type::Union{IndependentPopulationModel,SingleSessionPopulationModel} = IndependentPopulationModel(),
    verbose::Bool = true,
    kwargs...,
) where {prior_names,observation_names,action_names}

    #Check population_model
    check_population_model(
        IndependentPopulationModel(),
        action_model,
        prior,
        data,
        observation_cols,
        action_cols,
        session_cols,
        verbose;
        kwargs...,
    )

    #Get number of sessions
    n_sessions = length(groupby(data, session_cols))

    #Get the names of the estimated parameters
    parameters_to_estimate = keys(prior)

    #Create a filldist for each parameter
    priors_per_parameter = Tuple([
        filldist(getproperty(prior, param_name), n_sessions) for param_name in prior_names
    ])

    #Create a statistical model where the agents are independent and sampled from the same prior
    population_model =
        independent_population_model(priors_per_parameter, parameters_to_estimate)

    #Create a full model combining the agent model and the statistical model
    return create_model(
        action_model,
        population_model,
        data;
        observation_cols = observation_cols,
        action_cols = action_cols,
        session_cols = session_cols,
        parameters_to_estimate = parameters_to_estimate,
        population_model_type = population_model_type,
        kwargs...,
    )
end

#Turing model for sampling all sessions for all parameters
@model function independent_population_model(
    priors_per_parameter::T,
    parameters_to_estimate::Tuple{Vararg{Symbol}},
) where {T<:Tuple}

    sampled_parameters = Tuple(
        i ~ to_submodel(
            prefix(sample_parameters_all_session(prior), parameter_name),
            false,
        ) for
        (prior, parameter_name) in zip(priors_per_parameter, parameters_to_estimate)
    )

    #Slice, to allow for varying dimensionalities of parameters
    sampled_parameters = map(
        single_parameter ->
            single_parameter isa AbstractVector ? collect(single_parameter) :
            Array.(eachslice(single_parameter, dims = ndims(single_parameter))),
        sampled_parameters,
    )

    return zip(sampled_parameters...)
end

#Turing submodel for sampling all sessions for a single parameter
@model function sample_parameters_all_session(prior)

    session ~ prior

    return session
end



##############################################
####### CHECKS FOR THE POPULATION MODEL ######
##############################################
function check_population_model(
    model_type::IndependentPopulationModel,
    action_model::ActionModel,
    prior::NamedTuple{prior_names,<:Tuple{Vararg{Distribution}}},
    data::DataFrame,
    observation_cols::Union{
        NamedTuple{observation_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    session_cols::Union{Vector{Symbol},Symbol},
    verbose::Bool;
    kwargs...,
) where {prior_names,observation_names,action_names}
    #If there are no parameters to sample
    if length(prior) == 0
        #Throw an error
        throw(ArgumentError("There are no parameters in the prior."))
    end
end
