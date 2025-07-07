##################################
####### NO MISSING ACTIONS #######
##################################
function create_session_model(
    impute_missing_actions::NoMissingActions,    #No missing actions
    check_parameter_rejections::Val{false},     #No parameter rejections
    actions::Vector{Vector{AA}},
) where {A<:Any,AA<:Tuple{Vararg{Union{Array{A},A}}}}

    #Create flattened actions
    flattened_actions = evert(collect(Iterators.flatten(actions)))

    #Make multivariate actions into single arrays
    flattened_actions = Tuple(
        first(single_action) isa AbstractArray ?
        cat(single_action..., dims = ndims(first(single_action)) + 1) : single_action
        for single_action in flattened_actions
    )

    #Submodel for sampling all actions of a single action idx as an arraydist
    @model function sample_actions_one_idx(
        actions::Array{AA},
        distributions::Vector{D},
    ) where {A<:Any,AA<:Union{A,Array{A}},D<:Distribution}

        actions ~ arraydist(distributions)
    end

    #Create session model function with flattened actions included
    return @model function session_model(
        action_model::ActionModel,
        model_attributes::ModelAttributes,
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{O}},
        actions_per_session::Vector{Vector{AA}},
        estimated_parameter_names::Tuple{Vararg{Symbol}},
        session_ids::Vector{String};
        flattened_actions::Tuple{Vararg{Array{FA}}} = flattened_actions,
    ) where {
        O<:Tuple{Vararg{Any}},
        A<:Any,
        AA<:Tuple{Vararg{Union{Array{A},A}}},
        FA<:Union{Array{A},A},
        T,
    }

        ## Run forwards to get the action distributions ##
        action_distributions = [
            begin

                #Set the sampled parameters and reset the action model
                set_parameters!(
                    model_attributes,
                    estimated_parameter_names,
                    session_parameters,
                )
                reset!(model_attributes)
                [
                    begin
                        #Get the action probability (either a distribution, or a tuple of distributions) 
                        action_distribution =
                            action_model.action_model(model_attributes, observation...)

                        #Save the action (either a single real, or a tuple of reals)
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
        for (actions, distributions) in zip(flattened_actions, flattened_distributions)

            a ~ to_submodel(sample_actions_one_idx(actions, distributions), false)
        end
    end
end

####################################
####### WITH MISSING ACTIONS #######
####################################
function create_session_model(
    impute_missing_actions::AbstractMissingActions,
    check_parameter_rejections::Val{false},
    actions::Vector{Vector{AA}},
) where {A<:Union{Missing,Any},AA<:Tuple{Vararg{Union{Array{A},A}}}}

    ## Submodel for sampling all actions of a single action idx as an arraydist ##
    @model function sample_actions_one_idx(
        actions::Array{R},
        distributions::Vector{D},
    ) where {R<:Any,D<:Any}
        # ) where {R<:Real,D<:Distribution} TODO: once things have been made typestable we can clean this
        actions ~ arraydist(distributions)
    end

    ## Create timestep prefixes ##
    timestep_prefixes = [
        [Symbol(string("timestep_", j)) for j = 1:length(actions[i])] for
        i = 1:length(actions)
    ]

    ## Find timesteps with missing actions ##
    missing_action_markers = [
        Vector{AbstractMissingActions}(undef, length(session_actions)) for
        session_actions in actions
    ]
    #Go through each session
    for (session_idx, session_actions) in enumerate(actions)
        for (i, action) in enumerate(session_actions)

            #If it is missing, set the marker to the appropriate type
            if action isa Missing
                missing_action_markers[session_idx][i] = impute_missing_actions
            elseif action isa Tuple && any(x -> x isa Missing, action)
                missing_action_markers[session_idx][i] = impute_missing_actions
            else
                missing_action_markers[session_idx][i] = NoMissingActions()
            end
        end
    end

    ## Create flattened actions ##
    flattened_missing_action_markers = collect(Iterators.flatten(missing_action_markers))
    flattened_actions = collect(Iterators.flatten(actions))
    #Find the missing action markers
    flattened_actions = [
        marker isa NoMissingActions ? action : nothing for
        (action, marker) in zip(flattened_actions, flattened_missing_action_markers)
    ]
    #Filter out the missing actions and evert
    flattened_actions = filter(x -> !isnothing(x), flattened_actions)
    #Ensure type stability (TODO: once types are pre-inferred we can make this smarter)
    flattened_actions = [action for action in flattened_actions]
    #Make into a tuple of vectors
    flattened_actions = evert(flattened_actions)

    #Create session model function
    return @model function session_model(
        action_model::ActionModel,
        model_attributes::ModelAttributes,
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{O}},
        actions_per_session::Vector{Vector{AA}},
        estimated_parameter_names::Tuple{Vararg{Symbol}},
        session_ids::Vector{String};
        timestep_prefixes_per_session::Vector{Vector{Symbol}} = timestep_prefixes,
        missing_action_markers_per_session::Vector{Vector{AbstractMissingActions}} = missing_action_markers,
        flattened_actions::Tuple{Vararg{Array{FA}}} = flattened_actions,
    ) where {
        O<:Tuple{Vararg{Any}},
        A<:Union{Missing,Any},
        AA<:Tuple{Vararg{Union{Array{A},A}}},
        T<:Any,
        FA<:Union{Array{A},A},
    }
        ## Run forwards to get the action distributions ##
        action_distributions = [
            i ~ to_submodel(
                prefix(
                    single_session_model(
                        action_model,
                        model_attributes,
                        estimated_parameter_names,
                        session_parameters,
                        session_observations,
                        session_actions,
                        session_timestep_prefixes,
                        session_missing_action_markers,
                    ),
                    session_id,
                ),
                false,
            ) for (
                session_parameters,
                session_observations,
                session_actions,
                session_timestep_prefixes,
                session_missing_action_markers,
                session_id,
            ) in zip(
                parameters_per_session,
                observations_per_session,
                actions_per_session,
                timestep_prefixes_per_session,
                missing_action_markers_per_session,
                session_ids,
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
        for (actions, distributions) in zip(flattened_actions, flattened_distributions)
            a ~ to_submodel(sample_actions_one_idx(actions, distributions), false)
        end
    end
end


#######################################
####### SINGLE SESSION SUBMODEL #######
#######################################
@model function single_session_model(
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    estimated_parameter_names::Tuple{Vararg{Symbol}},
    session_parameters::T,
    session_observations::Vector{O},
    session_actions::Vector{AA},
    session_timestep_prefixes::Vector{Symbol},
    session_missing_action_markers::Vector{AbstractMissingActions},
) where {
    O<:Tuple{Vararg{Any}},
    A<:Union{Missing,Any},
    AA<:Tuple{Vararg{Union{Array{A},A}}},
    T<:Tuple,
}
    #Prepare the agent
    set_parameters!(model_attributes, estimated_parameter_names, session_parameters)
    reset!(model_attributes)

    return [
        i ~ to_submodel(
            prefix(
                sample_single_timestep(
                    action_model,
                    model_attributes,
                    observation,
                    action,
                    missing_action_marker,
                ),
                timestep_prefix,
            ),
            false,
        ) for (observation, action, timestep_prefix, missing_action_marker) in zip(
            session_observations,
            session_actions,
            session_timestep_prefixes,
            session_missing_action_markers,
        )
    ]
end

#########################################
####### SINGLE TIMESTEP SUBMODELS #######
#########################################
@model function sample_single_timestep(
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    observation::O,
    action::Tuple{Vararg{AA}},
    missing_actions::NoMissingActions,      #No missing actions
) where {O,A<:Any,AA<:Union{Array{A},A}}
    #Give observation and get action distribution
    action_distribution = action_model.action_model(model_attributes, observation...)

    #Store the action
    store_action!(model_attributes, action)

    #Return action distribution
    return action_distribution
end

@model function sample_single_timestep(
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    observation::O,
    action::Tuple{Vararg{Union{Missing,A,Array{A}}}},
    missing_actions::SkipMissingActions,    #Skip missing actions
) where {O,A<:Union{Missing,Any}}
    #Give observation and get action distribution
    action_distribution = action_model.action_model(model_attributes, observation...)

    #Return nothing
    return nothing
end

@model function sample_single_timestep(
    action_model::ActionModel,
    model_attributes::ModelAttributes,
    observation::O,
    actions::Tuple{Vararg{Union{Missing,A,Array{A}}}},
    missing_actions::InferMissingActions,   #Infer missing actions
) where {O,A<:Union{Missing,Any}}

    #Get the tuple of action distributions from the action model
    action_distributions = action_model.action_model(model_attributes, observation...)

    #TODO: Get rid of this when using pre-specified arrays
    if !(action_distributions isa Tuple)
        #If the action distributions are not a tuple, make them into a tuple
        action_distributions = (action_distributions,)
    end

    #Sample the actions separately from the probability distributions
    action = Tuple(
        i ~ to_submodel(
            prefix(sample_subaction(action, action_distribution), "$action_name"),
            false,
        ) for (action, action_distribution, action_name) in
        zip(actions, action_distributions, keys(action_model.actions))
    )

    #Store the action
    store_action!(model_attributes, action)

    return nothing
end

#Turing subsubmodel for sampling one of the actions in the timestep
@model function sample_subaction(
    action::Union{Missing,A,Array{A}},
    action_distribution::D,
) where {A<:Union{Any,Missing},D<:Distribution}
    action ~ action_distribution

    return action
end



####################################
####### PARAMETER REJECTIONS #######
####################################
function create_session_model(
    impute_missing_actions::AbstractMissingActions,
    check_parameter_rejections::Val{true},         #With parameter rejections
    actions::Vector{Vector{AA}},
) where {A<:Union{Any,Missing},AA<:Tuple{Vararg{Union{Array{A},A}}}}

    #Get normal session model
    session_submodel = create_session_model(impute_missing_actions, Val{false}(), actions)

    return @model function session_model(
        action_model::ActionModel,
        model_attributes::ModelAttributes,
        parameters_per_session::T, #No way to type for an iterator
        observations_per_session::Vector{Vector{O}},
        actions_per_session::Vector{Vector{AA}},
        estimated_parameter_names::Tuple{Vararg{Symbol}},
        session_ids::Vector{String},
    ) where {
        O<:Tuple{Vararg{Any}},
        A<:Union{Missing,Any},
        AA<:Tuple{Vararg{Union{Array{A},A}}},
        T<:Any,
    }

        try

            #Run the normal session model
            i ~ to_submodel(
                session_submodel(
                    action_model,
                    model_attributes,
                    parameters_per_session,
                    observations_per_session,
                    actions_per_session,
                    estimated_parameter_names,
                    session_ids,
                ),
                false,
            )

        catch e
            #If there is a parameter rejection, reject the sample
            if isa(e, RejectParameters)
                Turing.@addlogprob! -Inf
                return nothing
            else
                rethrow(e)
            end
        end
    end
end
