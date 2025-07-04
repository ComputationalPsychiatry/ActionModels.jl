##########################################################################################################
### FUNCTION FOR CREATING A CONDITIONED TURING MODEL FROM AN AGENT, A DATAFRAME AND A POPULATION MODEL ###
##########################################################################################################
"""
    create_model(action_model::ActionModel, population_model::DynamicPPL.Model, data::DataFrame; observation_cols, action_cols, session_cols=Vector{Symbol}(), parameters_to_estimate, impute_missing_actions=false, check_parameter_rejections=false, population_model_type=CustomPopulationModel(), verbose=true)

Create a `ModelFit` structure that can be used for sampling posterior and prior probability distributions. Consists of an action model, a population model, and a dataset.

This function prepares the data, checks consistency with the action and population models, handles missing data, and returns a `ModelFit` object ready for sampling and inference.

# Arguments
- `action_model::ActionModel`: The action model to fit.
- `population_model::DynamicPPL.Model`: The population model (e.g. a Turing model that generated parameters for each session).
- `data::DataFrame`: The dataset containing observations, actions, and session/grouping columns.
- `observation_cols`: Columns in `data` for observations. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `action_cols`: Columns in `data` for actions. Can be a `NamedTuple`, `Vector{Symbol}`, or `Symbol`.
- `session_cols`: Columns in `data` identifying sessions/groups (default: empty vector).
- `parameters_to_estimate`: Tuple of parameter names to estimate.
- `impute_missing_actions`: Whether to impute missing actions (default: `false`).
- `check_parameter_rejections`: Whether to check for parameter rejections (default: `false`).
- `population_model_type`: Type of population model (default: `CustomPopulationModel()`).
- `verbose`: Whether to print warnings and info (default: `true`).

# Returns
- `ModelFit`: Struct containing the model, data, and metadata for fitting and inference.

# Example
```jldoctest; setup = :(using ActionModels, DataFrames; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = @model function testmodel(population_args...) learning_rate ~ LogitNormal() end; population_model = population_model();)
julia> model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id, parameters_to_estimate = (:learning_rate,));

julia> model isa ActionModels.ModelFit
true
```

# Notes
- The returned `ModelFit` object can be used with `sample_posterior!`, `sample_prior!`, and other inference utilities.
- Handles missing actions according to the `impute_missing_actions` argument.
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
    impute_missing_actions::Bool = false,
    check_parameter_rejections::Bool = false,
    population_model_type::AbstractPopulationModel = CustomPopulationModel(),
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
    if !impute_missing_actions
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, collect(action_cols)]))
            #Remove any potential Missing type
            disallowmissing!(data, collect(action_cols))
            impute_missing_actions = NoMissingActions()
        else
            if verbose
                @warn """
                      There are missing values in the action columns, but impute_missing_actions is set to false. 
                      These actions will not be used to inform parameter estimation, and will be passed to the action model as missing values. 
                      Check that this is desired behaviour. This can be a problem for models which depend on their previous actions.
                      """
            end
            impute_missing_actions = SkipMissingActions()
        end
    else
        #If there are no missing actions
        if !any(ismissing, Matrix(data[!, collect(action_cols)]))
            if verbose
                @warn "impute_missing_actions is set to true, but there are no missing values in the action columns. Setting impute_missing_actions to false"
            end
            #Remove any potential Missing type
            disallowmissing!(data, collect(action_cols))
            impute_missing_actions = NoMissingActions()
        else
            impute_missing_actions = InferMissingActions()
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
    population_data = data[!,setdiff(Symbol.(names(data)), vcat(observation_cols, action_cols))]
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
    sessions_data = data[!, unique(vcat(collect(observation_cols), collect(action_cols), session_cols))]
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

    ### CREATE MODEL ###
    ## Create the session model ##
    session_model = create_session_model(
        impute_missing_actions,
        Val(check_parameter_rejections),
        actions,
    )

    ## Create a full model ##
    model = full_model(
        action_model,
        population_model,
        session_model,
        observations,
        actions,
        parameters_to_estimate,
        session_ids,
        initial_states,
    )

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



###########################
#### FULL TURING MODEL ####
###########################
@model function full_model(
    action_model::ActionModel,
    population_model::DynamicPPL.Model,
    session_model::Function,
    observations_per_session::Vector{Vector{O}},
    actions_per_session::Vector{Vector{AA}},
    estimated_parameter_names::Tuple{Vararg{Symbol}},
    session_ids::Vector{String},
    initial_states::NamedTuple{initial_state_keys,<:Tuple},
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
) where {
    O<:Tuple{Vararg{Any}},
    A<:Union{Missing,Real},
    AA<:Tuple{Vararg{Union{A,Array{A}}}},
    initial_state_keys,
    TF,
    TI,
}
    #Initialize the model with the correct types
    model_attributes = initialize_attributes(action_model, initial_states, TF, TI)

    #Generate session parameters with the population submodel
    parameters_per_session ~ to_submodel(population_model, false)

    #Generate behavior for each session
    i ~ to_submodel(
        session_model(
            action_model,
            model_attributes,
            parameters_per_session,
            observations_per_session,
            actions_per_session,
            estimated_parameter_names,
            session_ids,
        ),
        false, #Do not add a prefix
    )
end









#########################################
#### FUNCTION FOR CHECKING THE MODEL ####
#########################################
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
            if any(isnan.(skipmissing(data[!, colname])))
                throw(ArgumentError("There are NaN values in the action column $colname"))
            end
        end
    end
end
