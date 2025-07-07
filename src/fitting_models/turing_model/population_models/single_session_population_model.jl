#############################################
### POPULATION MODEL FOR A SINGLE SESSION ###
#############################################
"""
    create_model(action_model::ActionModel, prior::NamedTuple, observations::Vector, actions::Vector; verbose=true, kwargs...)

Create a Turing model for a single session with user-supplied observations and actions.

This function builds a model where all data belong to a single session, and parameters are sampled from the specified prior distributions. Returns a `ModelFit` object ready for sampling and inference.

# Arguments
- `action_model::ActionModel`: The agent/action model to fit.
- `prior::NamedTuple`: Named tuple of prior distributions for each parameter (e.g., `(; learning_rate = LogitNormal())`).
- `observations::Vector`: Vector of observations (or tuples of observations) for the session.
- `actions::Vector`: Vector of actions (or tuples of actions) for the session.
- `verbose`: Whether to print warnings and info (default: `true`).
- `kwargs...`: Additional keyword arguments passed to the underlying model constructor.

# Returns
- `ModelFit`: Struct containing the model, data, and metadata for fitting and inference.

# Example
```jldoctest; setup = :(using ActionModels; obs = [0.1, 0.2, 0.3, 0.4]; acts = [0.1, 0.2, 0.3, 0.4]; action_model = ActionModel(RescorlaWagner()); prior = (; learning_rate = LogitNormal()))
julia> model = create_model(action_model, prior, obs, acts); 

julia> model isa ActionModels.ModelFit
true
```

# Notes
- Use this model for fitting a single session or subject.
- The returned `ModelFit` object can be used with `sample_posterior!`, `sample_prior!`, and other inference utilities.
- Handles both scalar and tuple-valued observations/actions.
"""
function create_model(
    action_model::ActionModel,
    prior::NamedTuple{prior_names,<:Tuple{Vararg{Distribution}}},
    observations::Vector{OO},
    actions::Vector{AA};
    verbose::Bool = true,
    kwargs...,
) where {
    O<:Any,
    OO<:Union{O,Tuple{Vararg{O}}},
    A<:Union{Missing,Any},
    AA<:Union{A,Tuple{Vararg{Union{Missing,A}}}},
    prior_names,
}

    #Check population_model
    check_population_model(
        SingleSessionPopulationModel(),
        action_model,
        prior,
        observations,
        actions,
        verbose;
        kwargs...,
    )

    #Check if the observations and actions are tuples
    multiple_observations = OO <: Tuple
    multiple_actions = AA <: Tuple

    #Get number of observations and actions
    if !multiple_observations
        n_observations = 1
    else
        n_observations = length(first(observations))
    end
    if !multiple_actions
        n_actions = 1
    else
        n_actions = length(first(actions))
    end

    #Create column names
    observation_cols = map(x -> Symbol("observation_$x"), 1:n_observations)
    action_cols = map(x -> Symbol("action_$x"), 1:n_actions)

    #Make observations and actions into tuples of vectors
    if observations isa Vector{<:Tuple}
        observations = Tuple(
            [observation[i] for observation in observations] for
            i = 1:length(first(observations))
        )
    else
        observations = (observations,)
    end
    if actions isa Vector{<:Tuple}
        actions = Tuple([action[i] for action in actions] for i = 1:length(first(actions)))
    else
        actions = (actions,)
    end

    #Create dataframe of the observations and actions
    data = hcat(
        DataFrame(NamedTuple{Tuple(observation_cols)}(observations)),
        DataFrame(NamedTuple{Tuple(action_cols)}(actions)),
    )

    #Add session column
    session_cols = :session
    data[!, session_cols] .= :single_session

    #Create an independent_population_model with the single session
    return create_model(
        action_model,
        prior,
        data;
        observation_cols = observation_cols,
        action_cols = action_cols,
        session_cols = session_cols,
        verbose = verbose,
        population_model_type = SingleSessionPopulationModel(),
        kwargs...,
    )
end


##############################################
####### CHECKS FOR THE POPULATION MODEL ######
##############################################
function check_population_model(
    model_type::SingleSessionPopulationModel,
    action_model::ActionModel,
    prior::NamedTuple{prior_names,<:Tuple{Vararg{Distribution}}},
    observations::II,
    actions::AA,
    verbose::Bool;
    kwargs...,
) where {prior_names,II<:Vector,AA<:Vector}

    if length(observations) != length(actions)
        throw(
            ArgumentError(
                "The observations and actions vectors must have the same length.",
            ),
        )
    end
end




########################################
####### DEFAULT PLOTTING FUNCTION ######
########################################

#Plotting a ModelFit just plots the session parameters
@recipe function f(modelfit::ModelFit{SingleSessionPopulationModel}; plot_prior = true) #::Bool

    #Get session parameters
    posterior_parameters = get_session_parameters!(modelfit, :posterior)
    if plot_prior == true
        prior_parameters = get_session_parameters!(modelfit, :prior)
    else
        prior_parameters = nothing
    end

    #Get the session id
    session_id = modelfit.info.session_ids[1]

    #Make the standard plot for just that session
    plot(posterior_parameters, session_id; prior_parameters = prior_parameters)

end
