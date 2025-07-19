##########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A POPULATION MODEL ###
##########################################################################################################
"""
    create_model(action_model::ActionModel, population_model::DynamicPPL.Model, data::DataFrame; observation_cols, action_cols, session_cols=Vector{Symbol}(), parameters_to_estimate, missing_actions=NoMissingActions(), parameter_rejections=NoParameterChecking(), population_model_type=CustomPopulationModel(), verbose=true)

Create a `ModelFit` structure that can be used for sampling posterior and prior probability distributions. Consists of an action model, a population model, and a dataset.

This function prepares the data, checks consistency with the action and population models, handles missing data, and returns a `ModelFit` object ready for sampling and inference.

# Arguments
- `action_model::ActionModel`: The action model to fit.
- `population_model::DynamicPPL.Model`: The population model (e.g. a Turing model that generates parameters for each session).
- `data::DataFrame`: The dataset containing observations, actions, and session/grouping columns.
- `observation_cols`: Columns in `data` for observations. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `action_cols`: Columns in `data` for actions. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `session_cols`: Columns in `data` identifying sessions/groups (default: empty vector).
- `parameters_to_estimate`: Tuple of parameter names to estimate.
- `missing_actions`: Strategy for handling missing actions (default: `NoMissingActions()`). Use `SkipMissingActions()` to skip, or `InferMissingActions()` to infer missing actions.
- `parameter_rejections`: Strategy for handling parameter rejection errors (default: `NoParameterChecking()`). Use `ParameterChecking()` to enable rejecting samples.
- `population_model_type`: Type of population model (default: `CustomPopulationModel()`).
- `verbose`: Whether to print warnings and info (default: `true`).

# Returns
- `ModelFit`: Struct containing the model, data, and metadata for fitting and inference.

# Example
```jldoctest; setup = :(using ActionModels, DataFrames; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = @model function testmodel(population_args...) learning_rate ~ LogitNormal() end; population_model = population_model();)
julia> model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id, parameters_to_estimate = (:learning_rate,), missing_actions=NoMissingActions(), parameter_rejections=NoParameterChecking());

julia> model isa ActionModels.ModelFit
true
```

# Notes
- The returned `ModelFit` object can be used with `sample_posterior!`, `sample_prior!`, and other inference utilities.
- Handles missing actions according to the `missing_actions` argument.
- Checks for parameter rejection errors according to the `parameter_rejections` argument.
- Checks that columns and types in `data` match the action model specification.
"""
function create_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    data::DataFrame;
    observation_cols::Union{
        NamedTuple{observation_names_cols,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    action_cols::Union{
        NamedTuple{action_names_cols,<:Tuple{Vararg{Symbol}}},
        Vector{Symbol},
        Symbol,
    },
    session_cols::Union{Vector{Symbol},Symbol} = Vector{Symbol}(),
    parameters_to_estimate::Tuple{Vararg{Symbol}},
    missing_actions::AbstractMissingActions = NoMissingActions(),
    parameter_rejections::AbstractCheckParameterRejectionsMarker = NoParameterChecking(),
    population_model_type::AbstractPopulationModel = CustomPopulationModel(),
    sessions_model_type::AbstractSessionModel = FastSessionModel(),
    verbose::Bool = true,
) where {observation_names_cols,action_names_cols}

    ### ARGUMENT SETUP & CHECKS ###

    ## Change columns to the correct format ##
    #Make single action and observation columns into vectors
    if observation_cols isa Symbol
        observation_cols = [observation_cols]
    end
    if action_cols isa Symbol
        action_cols = [action_cols]
    end
    #Check that observation cols and action cols are the same length as the observations and actions in the action model
    if length(observation_cols) != length(action_model.observations)
        throw(
            ArgumentError(
                "The number of observation columns does not match the number of observations in the action model",
            ),
        )
    end
    if length(action_cols) != length(action_model.actions)
        throw(
            ArgumentError(
                "The number of action columns does not match the number of actions in the action model",
            ),
        )
    end

    #Make sure that observation_cols and action_cols are named tuples
    if observation_cols isa Vector
        observation_cols = NamedTuple{keys(action_model.observations)}(observation_cols)
        if verbose && length(observation_cols) > 1
            @warn "Mappings from action model observations to observation columns not provided. Using the order from the action model: $(observation_cols)"
        end
    end
    if action_cols isa Vector
        action_cols = NamedTuple{keys(action_model.actions)}(action_cols)
        if verbose && length(action_cols) > 1
            @warn "Mappings from action model actions to action columns not provided. Using the order from the action model: $(action_cols)"
        end
    end
    #Order observation and action columns to match the action model
    observation_cols = NamedTuple(
        observation_name => observation_cols[observation_name] for
        observation_name in keys(action_model.observations)
    )
    action_cols = NamedTuple(
        action_name => action_cols[action_name] for
        action_name in keys(action_model.actions)
    )

    #Grouping columns are a vector of symbols
    if !(session_cols isa Vector)
        session_cols = [session_cols]
    end

    ## Check whether to skip or infer missing data ##
    #If there are no missing actions
    if missing_actions isa NoMissingActions

        #If there are missing actions
        if any(ismissing, Matrix(data[!, collect(action_cols)]))
            throw(
                ArgumentError(
                    """
                    There are missing values in the action columns, but no strategy for handling them is specified.
                    Set missing_actions to SkipMissingActions() to skip these actions during model fitting, in which case they will be stored as missing values in the model attributes.
                    Set missing_actions to InferMissingActions() to treat these actions as latent variables and infer them during model fitting.
                    """,
                ),
            )

        else
            #Remove any potential Missing types in the data
            disallowmissing!(data, collect(action_cols))
        end
    end

    #If missing_actions are set to be skipped or inferred
    if (missing_actions isa SkipMissingActions || missing_actions isa InferMissingActions)
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, collect(action_cols)]))
            @warn "missing_actions is set to $missing_actions, but there are no missing actions in the dataset. Check that this is intended"
        end
    end

    ## Run checks for the model specifications ##
    check_model(
        action_model,
        population_model,
        data,
        observation_cols,
        action_cols,
        session_cols,
        population_model_type,
        parameters_to_estimate,
    )


    ### PREPARE DATA ###

    ## Extract action and observation types from the data ##
    observation_types_data = eltype.(eachcol(data[!, collect(observation_cols)]))
    action_types_data = eltype.(eachcol(data[!, collect(action_cols)]))

    ## Collect action names ##
    action_names = collect(keys(action_model.actions))

    ## Create a marker for multiple actions ##
    if length(action_names) > 1
        #Marker to indicate that there are multiple actions
        multiple_actions_marker = MultipleActions()
    else
        #Marker to indicate that there is only one action
        multiple_actions_marker = SingleAction()
    end

    ## Create list of initial states ##
    initial_state_parameter_state_names = NamedTuple(
        parameter.state => ParameterDependentState(parameter_name) for
        (parameter_name, parameter) in pairs(action_model.parameters) if
        parameter isa InitialStateParameter
    )
    initial_states = NamedTuple(
        state_name in keys(initial_state_parameter_state_names) ?
        state_name => initial_state_parameter_state_names[state_name] :
        state_name => state.initial_value for
        (state_name, state) in pairs(action_model.states)
    )

    ## Create population data ##
    #Remove action and observation columns
    population_data =
        data[!, setdiff(Symbol.(names(data)), vcat(observation_cols, action_cols))]
    #If there are session columns
    if length(session_cols) > 0
        #Only one row per session
        population_data = unique(population_data, session_cols)
        #Sort population data by session columns
        population_data = sort(population_data, session_cols)
    else
        #If there are no session columns, just take the first row
        population_data = population_data[1:1, :]
    end

    ## Create sessions data ##
    #Only keep actions, observations and session columns
    sessions_data =
        data[!, unique(vcat(collect(observation_cols), collect(action_cols), session_cols))]
    #Group sessions data by session columns
    sessions_data = groupby(sessions_data, session_cols, sort = true)

    ## Create IDs for each session ##
    session_ids = [
        join(
            [
                string(col_name) * id_column_separator * string(first(subdata)[col_name]) for col_name in session_cols
            ],
            id_separator,
        ) for subdata in sessions_data
    ]

    ## Extract observations and actions ##
    observations = Vector{Tuple{observation_types_data...}}[
        Tuple{observation_types_data...}.(
            eachrow(session_data[!, collect(observation_cols)]),
        ) for session_data in sessions_data
    ]
    actions = Vector{Tuple{action_types_data...}}[
        Tuple{action_types_data...}.(eachrow(session_data[!, collect(action_cols)])) for
        session_data in sessions_data
    ]


    ## Create missing action markers ##
    missing_action_markers = [
        begin
            #Get matrix with action cols
            submatrix = Matrix(session_subdata[:, collect(action_cols)])

            #Replace values with markers
            if missing_actions isa NoMissingActions

                #If there are no missing actions, set all markers to KnownAction
                submatrix =
                    map(x -> KnownAction(), submatrix)

            elseif missing_actions isa InferMissingActions
                #If missing actions should be inferred, set all missing actions to InferAction(), and the rest to KnownAction()
                submatrix = map(
                    x -> ifelse(ismissing(x), InferAction(), KnownAction()),
                    submatrix,
                )

            elseif missing_actions isa SkipMissingActions
                #If missing actions should be skipped, set all missing actions to SkipAction(), and the rest to KnownAction()
                submatrix = map(
                    x -> ifelse(ismissing(x), SkipAction(), KnownAction()),
                    submatrix,
                )
            end

            #Convert into a vector with a tuple for each row (timestep)
            Vector{AbstractMissingActionMarker}.(eachrow(submatrix))
        end

        #For each session
        for session_subdata in sessions_data
    ]


    ### PREPARE SESSIONS MODEL TYPE ###
    if sessions_model_type isa FastSessionModel

        #Prepare flattened actions and missing action markers
        flattened_actions = prepare_flattened_actions(
            actions,
            missing_action_markers,
        )

        #Create the sessions model type with flattened actions
        sessions_model_type = FastSessionModel(flattened_actions, missing_actions)
    end

    ### CREATE FULL MODEL ###
    #Create the Turing model
    model = full_model(
        action_model,
        population_model,
        initial_states,
        parameters_to_estimate,
        action_names,
        parameter_rejections,
        sessions_model_type,
        session_ids,
        observations,
        actions,
        missing_action_markers,
        multiple_actions_marker,
    )

    #Return it as part of a ModelFit
    return ModelFit(
        model = model,
        population_model_type = population_model_type,
        population_data = population_data,
        info = ModelFitInfo(
            estimated_parameter_names = parameters_to_estimate,
            session_ids = session_ids,
        ),
    )
end




################################
#### OUTERMOST TURING MODEL ####
################################
@model function full_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    initial_states::NamedTuple{initial_state_keys,<:Tuple},
    parameter_names::Tuple{Vararg{Symbol}},
    action_names::Vector{Symbol},
    check_rejections_marker::AbstractCheckParameterRejectionsMarker,
    sessions_model_type::AbstractSessionModel,
    session_ids::Vector{String},
    observations::Vector{Vector{OO}},
    actions::Vector{Vector{AA}},
    missing_action_markers::Vector{Vector{MM}},
    multiple_actions_marker::AbstractMultipleActionsMarker,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {
    initial_state_keys,
    O,
    OO<:Tuple{Vararg{O}},
    A,
    AA<:Tuple{Vararg{A}},
    MM<:Vector{AbstractMissingActionMarker},
    TF,
    TI,
}
    #Initialize the model with the correct types
    model_attributes = initialize_attributes(action_model, initial_states, TF, TI)

    #Generate session parameters with the population submodel
    parameters ~ to_submodel(population_model, false)

    #Generate behavior for each session
    i ~ to_submodel(
        sessions_model(
            check_rejections_marker,
            sessions_model_type,
            action_model,
            model_attributes,
            session_ids,
            parameter_names,
            action_names,
            parameters,
            observations,
            actions,
            missing_action_markers,
            multiple_actions_marker,
        ),
        false, #Do not add a prefix
    )
end




#############################################
#### FUNCTION FOR CHECKING THE ARGUMENTS ####
#############################################
function check_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    data::DataFrame,
    observation_cols::NamedTuple{observation_names_cols,<:Tuple{Vararg{Symbol}}},
    action_cols::NamedTuple{action_names_cols,<:Tuple{Vararg{Symbol}}},
    session_cols::Vector{Symbol},
    population_model_type::AbstractPopulationModel,
    parameters_to_estimate::Tuple{Vararg{Symbol}},
) where {observation_names_cols,action_names_cols}

    #Check that user-specified columns exist in the dataset
    if any(session_cols .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified group columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(observation_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified observation columns that do not exist in the dataframe",
            ),
        )
    elseif any(values(action_cols) .∉ Ref(Symbol.(names(data))))
        throw(
            ArgumentError(
                "There are specified action columns that do not exist in the dataframe",
            ),
        )
    end

    #Check that observation and action column names exist in the action model
    for (observation_name_col, observation_col) in pairs(observation_cols)
        if !(observation_name_col in keys(action_model.observations))
            throw(
                ArgumentError(
                    "The observation column $observation_col does not exist in the action model",
                ),
            )
        end
    end
    for (action_name_col, action_col) in pairs(action_cols)
        if !(action_name_col in keys(action_model.actions))
            throw(
                ArgumentError(
                    "The action column $action_col does not exist in the action model",
                ),
            )
        end
    end

    #Check whether observation and action columns are subtypes of what is specified in the action model
    for (action_col, (action_name, action)) in zip(action_cols, pairs(action_model.actions))
        if !(
            eltype(data[!, action_col]) <: action.type ||
            eltype(data[!, action_col]) <: Union{Missing,T} where {T<:action.type}
        )
            throw(
                ArgumentError(
                    "The action colum $action_col has type $(eltype(data[!, action_col])), but must be a subtype of the $action_name type specified in the action model: $(action.type)",
                ),
            )
        end
    end
    for (observation_col, (observation_name, observation)) in
        zip(observation_cols, pairs(action_model.observations))
        if !(eltype(data[!, observation_col]) <: observation.type)
            throw(
                ArgumentError(
                    "The observation column $observation_col has type $(eltype(data[!, observation_col])), but must be a subtype of the $observation_name type specified in the action model: $(observation.type)",
                ),
            )
        end
    end

    #Check whether there are NaN values in the action columns
    for (colname, action) in zip(action_cols, action_model.actions)
        if action.type <: AbstractArray
            if any(
                map(
                    action_array -> any(isnan.(skipmissing((action_array)))),
                    data[!, colname],
                ),
            )
                throw(ArgumentError("There are NaN values in the action column $colname"))
            end
        else
            # if any(isnan.(skipmissing(data[!, colname])))
            #     throw(ArgumentError("There are NaN values in the action column $colname"))
            # end
        end
    end
end
