#########################################################
####### FUNCTION FOR EXTRACTING GENERATED QUANTITIES ####
#########################################################
function extract_quantities(model::DynamicPPL.Model, fitted_model::Chains)

    @show typeof(fitted_model), typeof(model)
    # Extract the generated quantities from the fitted model
    quantities = generated_quantities(model, fitted_model)

    # Extract agent ids
    agent_ids = model.args.agent_ids
    n_agents = length(agent_ids)

    # Extract parameter keys
    parameter_keys = collect(keys(first(first(quantities).agent_parameters)))
    parameter_keys_symbols = [
        begin
            if parameter_key isa Tuple
                Symbol(join(parameter_key, tuple_separator))
            else
                Symbol(parameter_key)
            end
        end for parameter_key in parameter_keys
    ]

    # Get the dimensionality of the output array
    (n_samples, _, n_chains) = size(fitted_model.value)
    n_agents = length(agent_ids)
    n_parameters = length(parameter_keys)

    # Create an empty AxisArray
    empty_array = Array{Float64}(undef, n_agents, n_parameters, n_samples, n_chains)
    parameter_values = AxisArray(
        empty_array,
        Axis{:agent}(agent_ids),
        Axis{:parameter}(parameter_keys_symbols),
        Axis{:sample}(1:n_samples),
        Axis{:chain}(1:n_chains),
    )

    # For each chain and each sample
    for chain_idx = 1:n_chains
        for sample_idx = 1:n_samples

            #Extract the corresponding quantity.
            sample_quantities = quantities[(chain_idx-1)*n_samples+sample_idx]
            sample_agent_parameters = sample_quantities.agent_parameters

            # For each agent
            for (agent_idx, agent_id) in enumerate(agent_ids)
                agent_parameters = sample_agent_parameters[agent_idx]

                # For each aprameter
                for parameter_key in parameter_keys
                    # Join tuples
                    if parameter_key isa Tuple
                        parameter_key_symbol = Symbol(join(parameter_key, tuple_separator))
                    else
                        parameter_key_symbol = Symbol(parameter_key)
                    end
                    #Store the value
                    parameter_values[
                        agent_idx,
                        parameter_key_symbol,
                        sample_idx,
                        chain_idx,
                    ] = agent_parameters[parameter_key]
                end
            end
        end
    end

    #Extract the other values from the population model
    other_values = [quantity.other_values for quantity in quantities]
    #If they are all nothing, shorten them to one nothing
    if all(isnothing, other_values)
        other_values = nothing
    end

    #Only return other values if there are any
    if isnothing(other_values)
        return parameter_values
    else
        return (agent_parameters = parameter_values, other_values = other_values)
    end
end
