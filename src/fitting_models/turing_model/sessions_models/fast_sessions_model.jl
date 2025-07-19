#########################################
####### Prepare flattened actions #######
#########################################
#Function which prepares the flattened actions for the sessions model
function prepare_flattened_actions(
    actions::Vector{Vector{AA}},
    missing_action_markers::Vector{Vector{MM}},
) where {A<:Any, AA<:Tuple{Vararg{Union{Array{A},A}}}, M<:AbstractMissingActionMarker, MM<:Vector{M}}

    ## Create flattened actions ##
    flattened_missing_action_markers = collect(Iterators.flatten(missing_action_markers))
    flattened_actions = collect(Iterators.flatten(actions))
    #Find the missing action markers
    flattened_actions = [
        all(typeof.(marker) .<: KnownAction) ? action : nothing for
        (action, marker) in zip(flattened_actions, flattened_missing_action_markers)
    ]
    #Filter out the missing actions and evert
    flattened_actions = filter(x -> !isnothing(x), flattened_actions)
    #Ensure type stability (TODO: once types are pre-inferred we can make this smarter)
    flattened_actions = [action for action in flattened_actions]
    #Make into a tuple of vectors
    flattened_actions = evert(flattened_actions)

    return flattened_actions
end


###################################################################
####### SUBMODEL FOR FITTING FULL SET OF SINGLE ACTION TYPE #######
###################################################################
@model function sample_actions_one_type(
    actions::Array{AA},
    distributions::Vector{D},
) where {A<:Any,AA<:Union{A,Array{A}},D<:Distribution}
    #Sample all the actions of one type as a product distribution
    actions ~ product_distribution(distributions...)
end



##########################################
####### NO MISSING ACTIONS VARIANT #######
##########################################
@model function sessions_model(
    check_rejections_marker::NoParameterChecking,
    sessions_model_type::FastSessionModel{NoMissingActions},
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    session_ids::Vector{String},
    parameter_names::Tuple{Vararg{Symbol}},
    action_names::Vector{Symbol},
    parameters_per_session::PP, #Ducktyping: must be iterator over Tuples of Real or Array{Real}
    observations_per_session::Vector{Vector{OO}},
    actions_per_session::Vector{Vector{AA}},
    missing_action_markers::Vector{Vector{MM}},
    multiple_actions_marker::AbstractMultipleActionsMarker,
) where {
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    PP,
    MM<:Vector{AbstractMissingActionMarker},
}
    ## Run forwards to get the action distributions ##
    action_distributions = [
        begin

            #Set the sampled parameters and reset the action model
            set_parameters!(model_attributes, parameter_names, session_parameters)
            reset!(model_attributes)
            [
                begin
                    #Get the action probability (either a distribution, or a tuple of distributions) 
                    action_distribution =
                        action_model.action_model(model_attributes, observation...)

                    #Save the action (either a single action, or a tuple of actions)
                    store_action!(model_attributes, action)

                    #Return the action probability distribution
                    action_distribution

                end for
                (observation, action) in zip(session_observations, session_actions)
            ]
        end for (session_parameters, session_observations, session_actions) in
        zip(parameters_per_session, observations_per_session, actions_per_session)
    ]

    ## Reshape into a tuple of vectors with distributions ## 
    flattened_distributions = evert(collect(Iterators.flatten(action_distributions)))

    ## Sample the actions from the probability distributions ##
    for (actions_single_type, distributions) in
        zip(sessions_model_type.flattened_actions, flattened_distributions)
        a ~ to_submodel(sample_actions_one_type(actions_single_type, distributions), false)
    end
end




#######################################
####### MISSING ACTIONS VARIANT #######
#######################################
## Outer-layer sessions model ##
@model function sessions_model(
    check_rejections_marker::NoParameterChecking,
    sessions_model_type::FastSessionModel,
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

    ## Run forwards to get the action distributions for each session ##
    action_distributions = [

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

        #For each session s
        for
        (session_idₛ, parametersₛ, observationsₛ, actionsₛ, missing_action_markersₛ) in
        zip(
            session_ids,
            parameters_per_session,
            observations_per_session,
            actions_per_session,
            missing_action_markers_per_session,
        )
    ]

    #Remove missing action distributions
    flattened_distributions =
        filter(x -> !isnothing(x), collect(Iterators.flatten(action_distributions)))

    #Ensure type stability (TODO: once types are pre-inferred we can make this smarter)
    flattened_distributions = [distribution for distribution in flattened_distributions]
    #Make distributions into a tuple of vectors
    flattened_distributions = evert(flattened_distributions)

    ## Sample the actions from the probability distributions ##
    for (actions_single_type, distributions) in zip(sessions_model_type.flattened_actions, flattened_distributions)
        a ~ to_submodel(sample_actions_one_type(actions_single_type, distributions), false)
    end
end


## Single session ##
@model function single_session(
    sessions_model_type::FastSessionModel,
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

    #Get flattened actions
    return [
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
            #For each timestep t
        ) for (t, (observationsₜ, actionsₜ, missing_action_markersₜ)) in
        enumerate(zip(observationsₛ, actionsₛ, missing_action_markersₛ))
    ]
end


## Single timestep, all actions known ##
@model function single_timestep(
    sessions_model_type::FastSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    action_names::Vector{Symbol},
    observationsₜ::OO,
    actionsₜ::AA,
    missing_action_markersₜ::MM,
    multiple_actions_marker::AbstractMultipleActionsMarker,
) where {O,OO<:Tuple{Vararg{O}},A,AA<:Tuple{Vararg{A}},MM<:Vector{KnownAction}}
    #Give observation and get action distribution
    action_distribution = action_model.action_model(model_attributes, observationsₜ...)

    #Store the action
    store_action!(model_attributes, actionsₜ...)

    #Return action distribution
    return action_distribution
end



## Single timestep, some unknown actions ##
@model function single_timestep(
    sessions_model_type::FastSessionModel,
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    action_names::Vector{Symbol},
    observationsₜ::OO,
    actionsₜ::AA,
    missing_action_markersₜ::MM,
    multiple_actions_marker::AbstractMultipleActionsMarker,
) where {
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    MM<:Vector{AbstractMissingActionMarker},
}
    #Use the pointwise sessions model to fit the actions of the single timestep
    i ~ to_submodel(
        prefix(
            single_timestep(
                PointwiseSessionModel(),
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

    #Return nothing
    return nothing
end