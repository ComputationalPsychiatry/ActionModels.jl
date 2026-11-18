#######################################
##### SUMMARIZE SESSION PARAMETERS ####
#######################################
"""
    MCMCChains.summarize(session_parameters::SessionParameters, summary_function::Function=median)

Summarize posterior samples of session-level parameters into a tidy `DataFrame`.

Each row corresponds to a session, and each column to a parameter (or parameter element, for arrays). The summary statistic (e.g., `median`, `mean`) is applied to the posterior samples for each parameter and session.

# Arguments
- `session_parameters::SessionParameters`: Posterior samples of session-level parameters, as returned by model fitting.
- `summary_function::Function=median`: Function to summarize the samples (e.g., `mean`, `median`, `std`).

# Returns
- `DataFrame`: Table with one row per session and columns for each parameter (or parameter element).

# Example
```jldoctest; setup = :(using ActionModels, DataFrames, StatsPlots; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = (; learning_rate = LogitNormal()); model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id); chns = sample_posterior!(model, sampler = HMC(0.8, 10),n_samples=100, n_chains=1, progress = false))
julia> df = summarize(get_session_parameters!(model), median);

julia> df isa DataFrame
true
```

# Notes
- For array-valued parameters, columns are named with indices, e.g., `:expected_value[1]`.
- Session identifiers are split into columns if composite.
"""
function MCMCChains.summarize(
    session_parameters::SessionParameters,
    summary_function::Function = median;
)
    #Extract sessions and parameters
    session_ids = session_parameters.session_ids
    estimated_parameter_names = session_parameters.estimated_parameter_names
    parameter_types = session_parameters.parameter_types
    session_parameters = session_parameters.value

    #Construct session column names
    session_cols = [
        Symbol(first(split(i, id_column_separator))) for
        i in split(string(first(session_ids)), id_separator)
    ]

    ## Initialize an empty DataFrame with appropriate columns. ##
    df_cols = Dict{String,Any}()
    parameter_colnames = Dict{String,Vector{String}}()
    # For each ParameterDependentState
    for parameter in estimated_parameter_names
        #If its first sample is an AbstractArray
        sample_parameter = first(session_parameters[parameter][1])
        if sample_parameter isa AbstractArray
            #Make container for the column names
            names = String[]
            #Get the parameter type
            parameter_type = eltype(parameter_types[parameter])
            #For each index in the parameter
            for I in CartesianIndices(size(sample_parameter))
                #Construct the column name
                col_name = "$(parameter)$(collect(Tuple(I)))"
                #Add it to the list of column names
                push!(names, col_name)
                #Add it to the dataframe column names with the appropriate type
                df_cols[col_name] = parameter_type[]
            end
            #Save the column names
            parameter_colnames["$parameter"] = names
        else
            #Oherwise, just make a column for the state
            df_cols["$parameter"] = parameter_types[parameter][]
        end
    end

    #Create dataframe
    df = DataFrame(df_cols)

    #Add session colnames
    for column_name in session_cols
        df[!, column_name] = String[]
    end

    ## Populate the DataFrame with summarized values ##
    row = Dict()
    #Loop over sessions and parameters
    for session_id in session_ids
        for parameter in estimated_parameter_names

            #Split session ids
            split_session_ids = split(string(session_id), id_separator)
            #Add them to the row
            for (session_id_part, column_name) in zip(split_session_ids, session_cols)
                row["$column_name"] = string(split(session_id_part, id_column_separator)[2])
            end

            # Extract the values for the current session and parameter across samples and chains
            samples = session_parameters[parameter][Symbol(session_id)]

            #If the samples are AbstractArrays
            if first(samples) isa AbstractArray

                #Get column names for the parameter
                colnames = parameter_colnames["$parameter"]

                #For each Cartesian index and corresponding column name
                for (I, colname) in zip(CartesianIndices(size(first(samples))), colnames)
                    #Unpack that index and summarize the samples
                    summarized_value =
                        summarize_samples(map(s -> s[I], samples), summary_function)
                    row[colname] = summarized_value
                end
            else
                summarized_value = summarize_samples(samples, summary_function)
                row["$parameter"] = summarized_value
            end
        end
        # Add the row to the DataFrame
        push!(df, row, promote = true)
    end

    # Reorder the columns to have session id's as the first columns
    select!(df, session_cols, names(df)[1:(end-length(session_cols))]...)

    return df
end


#######################################
##### SUMMARIZE STATE TRAJECTORIES ####
#######################################
"""
    MCMCChains.summarize(state_trajectories::StateTrajectories, summary_function::Function=median)

Summarize posterior samples of state trajectories into a tidy `DataFrame`.

Each row corresponds to a session and timestep, and each column to a state variable (or state element, for arrays). The summary statistic (e.g., `median`, `mean`) is applied to the posterior samples for each state, session, and timestep.

# Arguments
- `state_trajectories::StateTrajectories`: Posterior samples of state trajectories, as returned by model fitting.
- `summary_function::Function=median`: Function to summarize the samples (e.g., `mean`, `median`, `std`).

# Returns
- `DataFrame`: Table with one row per session and timestep, and columns for each state (or state element).

# Example
```jldoctest; setup = :(using ActionModels, DataFrames, StatsPlots; data = DataFrame("id" => ["S1", "S1", "S2", "S2"], "observation" => [0.1, 0.2, 0.3, 0.4], "action" => [0.1, 0.2, 0.3, 0.4]); action_model = ActionModel(RescorlaWagner()); population_model = (; learning_rate = LogitNormal()); model = create_model(action_model, population_model, data; action_cols = :action, observation_cols = :observation, session_cols = :id); chns = sample_posterior!(model, sampler = HMC(0.8, 10),n_samples=100, n_chains=1, progress = false))
julia> df = summarize(get_state_trajectories!(model, :expected_value), mean);

julia> df isa DataFrame
true
```

# Notes
- For array-valued states, columns are named with indices, e.g., `:expected_value[1]`.
- Session identifiers are split into columns if composite.
- The column `timestep` indicates the time index (starting from 0).
"""
function MCMCChains.summarize(
    state_trajectories::StateTrajectories,
    summary_function::Function = median,
)

    #Extract sessions ids and state trajectories
    state_names = state_trajectories.state_names
    session_ids = state_trajectories.session_ids
    state_types = state_trajectories.state_types
    state_trajectories = state_trajectories.value

    #Construct session column names
    session_cols = [
        first(split(i, id_column_separator)) for
        i in split(string(first(session_ids)), id_separator)
    ]

    ## Initialize an empty DataFrame with appropriate columns. ##
    df_cols = Dict{String,Any}()
    state_colnames = Dict{String,Vector{String}}()
    # For each state
    for state in state_names
        #If its first sample is an AbstractArray
        sample_state = first(state_trajectories[state][1])
        if sample_state isa AbstractArray
            names = String[]
            state_type = eltype(state_types[state])
            #Make a column for each Cartesian index
            for I in CartesianIndices(size(sample_state))
                #Construct the column name
                col_name = "$(state)$(collect(Tuple(I)))"
                #Add it to the list of column names
                push!(names, col_name)
                #Add it to the dataframe column names with the appropriate type
                df_cols[col_name] = state_type[]
            end
            #Save the column names
            state_colnames["$state"] = names
        else
            #Oherwise, just make a column for the state
            df_cols["$state"] = state_types[state][]
        end
    end
    #Create dataframe
    df = DataFrame(df_cols)

    #Add session columns and the timestep column.
    for column_name in session_cols
        df[!, column_name] = String[]
    end
    df[!, "timestep"] = Int[]


    # Populate the DataFrame with summarized values
    row = Dict()
    #Loop for sessions and states
    for session_id in session_ids
        for state in state_names

            #Split session ids
            split_session_ids = split(string(session_id), id_separator)
            #Add them to the row
            for (session_id_part, column_name) in zip(split_session_ids, session_cols)
                row["$column_name"] = string(split(session_id_part, id_column_separator)[2])
            end

            # Extract the values for the current session and state across samples, chains and timesteps
            values = state_trajectories[state][Symbol(session_id)]

            #Loop over timesteps
            for timestep = 1:size(values, 3)

                # Add the timestep to the row
                row["timestep"] = timestep-1

                # Extract the values for the current session and state across samples and chains
                samples = values[:, :, timestep]

                #If the samples are AbstractArrays
                if first(samples) isa AbstractArray
                    #Get column names for the state
                    colnames = state_colnames["$state"]

                    #For each Cartesian index and corresponding column name
                    for (I, colname) in
                        zip(CartesianIndices(size(first(samples))), colnames)
                        #Unpack that index and summarize the samples
                        summarized_value =
                            summarize_samples(map(s -> s[I], samples), summary_function)
                        row[colname] = summarized_value
                    end
                else
                    summarized_value = summarize_samples(samples, summary_function)
                    row["$state"] = summarized_value
                end

                # Add the row to the DataFrame
                push!(df, row; promote = true)
            end
        end
    end

    # Reorder the columns to have session id's as the first columns
    select!(df, session_cols, "timestep", names(df)[1:(end-length(session_cols)-1)]...)

    return df
end

function summarize_samples(array::A, summary_function::Function) where {A<:AxisArray}
    # Handle missing values
    if all(ismissing, array)
        return missing
    elseif any(ismissing, array)
        return summary_function(skipmissing(array))
    else
        return summary_function(array)
    end
end
