### Turing submodel for all sessions ###
@model function sessions_model(
    check_rejections_marker::NoParameterChecking,
    sessions_model_type::PointwiseSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    session_ids::Vector{String},
    parameter_names::Tuple{Vararg{Symbol}},
    action_names::Vector{Symbol},
    parameters_per_session::PP, #Ducktyping: must be iterator over Tuples of Real or Array{Real}
    observations_per_session::Vector{Vector{OO}},
    actions_per_session::Vector{Vector{AA}},
    missing_action_markers_per_session::Vector{Vector{MM}},
    multiple_actions_marker::AbstractMultipleActionsMarker,
) where {
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    PP,
    MM<:Vector{AbstractMissingActionMarker},
}

    #For each session s
    for (session_idₛ, parametersₛ, observationsₛ, actionsₛ, missing_action_markersₛ) in zip(
        session_ids,
        parameters_per_session,
        observations_per_session,
        actions_per_session,
        missing_action_markers_per_session,
    )

        #Run the single session submodel
        i ~ to_submodel(
            prefix(
                single_session(
                    sessions_model_type,
                    action_model,
                    model_attributes,
                    parameter_names,
                    action_names,
                    parametersₛ,
                    observationsₛ,
                    actionsₛ,
                    missing_action_markersₛ,
                    multiple_actions_marker,
                ),
                session_idₛ,
            ),
            false,
        )
    end
end

### Turing submodel for running a single session ###
@model function single_session(
    sessions_model_type::PointwiseSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    parameter_names::Tuple{Vararg{Symbol}},
    action_names::Vector{Symbol},
    parametersₛ::PP,
    observationsₛ::Vector{OO},
    actionsₛ::Vector{AA},
    missing_action_markersₛ::Vector,#{MM},
    multiple_actions_marker::AbstractMultipleActionsMarker,
) where {
    P<:Real,
    PP<:Tuple{Vararg{Union{P,Array{P}}}},
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    # M<:AbstractMissingActionMarker,
    # MM<:Vector{M},
}
    #Set the sampled parameters
    set_parameters!(model_attributes, parameter_names, parametersₛ)
    #Reset the model attributes
    reset!(model_attributes)

    #For each timestep t
    for (t, (observationsₜ, actionsₜ, missing_action_markersₜ)) in
        enumerate(zip(observationsₛ, actionsₛ, missing_action_markersₛ))

        #Run single timestep submodel
        i ~ to_submodel(
            prefix(
                single_timestep(
                    sessions_model_type,
                    action_model,
                    model_attributes,
                    action_names,
                    observationsₜ,
                    actionsₜ,
                    missing_action_markersₜ,
                    multiple_actions_marker,
                ),
                t,
            ),
            false,
        )
    end
end

### Turing submodel for running a single timestep ###
## Single action ##
@model function single_timestep(
    sessions_model_type::PointwiseSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    action_names::Vector{Symbol},
    observationsₜ::OO,
    actionsₜ::AA,
    missing_action_markersₜ::MM,
    multiple_actions_marker::SingleAction,
) where {
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    MM<:Vector{AbstractMissingActionMarker},
}

    #Give observation and get the action distribution
    action_distributionₜ = action_model.action_model(model_attributes, observationsₜ...)

    #Sample the action (unpack action, missing_action_marker, and action_name)
    sampled_actionₜ ~ to_submodel(
        prefix(
            single_action(
                first(actionsₜ),
                action_distributionₜ,
                first(missing_action_markersₜ),
            ),
            first(action_names),
        ),
        false,
    )


    #Store the actions
    store_action!(model_attributes, sampled_actionₜ)

    return nothing
end

## Multiple actions ##
@model function single_timestep(
    sessions_model_type::PointwiseSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    action_names::Vector{Symbol},
    observationsₜ::OO,
    actionsₜ::AA,
    missing_action_markersₜ::MM,
    multiple_actions_marker::MultipleActions,
) where {
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    MM<:Vector{AbstractMissingActionMarker},
}

    #Give observation and get action distributions
    action_distributionsₜ = action_model.action_model(model_attributes, observationsₜ...)

    #Sample each action a
    sampled_actionsₜ = AA(
        i ~ to_submodel(
            prefix(
                single_action(actionₐ, action_distributionₐ, missing_action_markerₐ),
                action_nameₐ,
            ),
            false,
        ) for (actionₐ, action_distributionₐ, missing_action_markerₐ, action_nameₐ) in
        zip(actionsₜ, action_distributionsₜ, missing_action_markersₜ, action_names)
    )

    #Store the actions
    store_action!(model_attributes, sampled_actionsₜ)

    return nothing
end

### Turing submodel for sampling a single action ###
## Known action ##
@model function single_action(
    action::A,
    action_distribution::D,
    missing_action::KnownAction,
) where {A,D<:Distribution}

    #Sample the action (i.e., calculate the logprob)
    action ~ action_distribution

    return action
end

## Infer action ##
@model function single_action(
    action::A,
    action_distribution::D,
    missing_action::InferAction,
) where {A,D<:Distribution}

    #Sample the action
    action ~ action_distribution

    return action
end

## Skip action ##
@model function single_action(
    action::A,
    action_distribution::D,
    missing_action::SkipAction,
) where {A,D<:Distribution}

    # Just return a missing as the action
    return missing
end