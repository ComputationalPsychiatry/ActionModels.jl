## GET TYPES ##
function ActionModels.get_parameter_types(multisubmodel::MultiSubmodel)
    #create empty map
    parameter_map = (;)

    #going through all submodels 
    for (submodel_name, submodel) in pairs(multisubmodel.submodels)

        #going through the parameter names and getting their actual value
        for (parameter_name, value) in pairs(ActionModels.get_parameter_types(submodel))

            merged_name = Symbol(string(submodel_name) * "_" * string(parameter_name))
            #merging the name with the value
            parameter_map = merge(parameter_map, NamedTuple{(merged_name,)}((value, )))

        end
       
    end

    return parameter_map

end

function ActionModels.get_state_types(multisubmodel::MultiSubmodel)
    #create empty map
    state_map = (;)
    
    #going through all submodels 
    for (submodel_name, submodel) in pairs(multisubmodel.submodels)
        
        #going through the states and getting their value
        for (state_name, value) in pairs(ActionModels.get_state_types(submodel))

            merged_name = Symbol(string(submodel_name) * "_" * string(state_name))
            #merging names and values of the states
            state_map = merge(state_map, NamedTuple{(merged_name,)}((value, )))

        end
       
    end

    return state_map

end


function ActionModels.initialize_attributes(
    multisubmodel::MultiSubmodel,
    ::Type{TF} = Float64,
    ::Type{TI} = Int64,
    ) where {TF,TI}
    
    # initialize attributes for all models
    attributes_submodels = map(
        submodel -> ActionModels.initialize_attributes(submodel, TF, TI),
        multisubmodel.submodels
    )

    return MultiSubmodelAttributes(attributes_submodels, multisubmodel.parameter_map, multisubmodel.state_map)
end


## MULITSUBMODEL ATTRIBUTE API ##
function ActionModels.reset!(multisubmodel_attributes::MultiSubmodelAttributes)
    
    #Reset all submodels
    map(attributes_submodel -> reset!(attributes_submodel), multisubmodel_attributes.attributes_submodels)
    
    return nothing
end

#Get all parameters and states
function ActionModels.get_parameters(multisubmodel_attributes::MultiSubmodelAttributes)
    
    #extract the submodel name -> tub[1] and submodel itself tub [2] 
    #map the parameters to their values
    return map(
        tup -> 
            get_parameters(getfield(multisubmodel_attributes.attributes_submodels, tup[1]), tup[2]), 
        multisubmodel_attributes.parameter_map)
end

function ActionModels.get_states(multisubmodel_attributes::MultiSubmodelAttributes)
   
    #extract the submodel name -> tub[1] and submodel itself tub [2] 
    #map the states to their values
    return map(
        tup -> 
            get_states(getfield(multisubmodel_attributes.attributes_submodels, tup[1]), tup[2]),
        multisubmodel_attributes.state_map)
end

#Get specific parameter or state
function ActionModels.get_parameters(
    multisubmodel_attributes::MultiSubmodelAttributes,
    merged_parameter_name::Symbol,
)
    #Check if parameter name is in parameter map, if not return AttributeError
    if !(merged_parameter_name in keys(multisubmodel_attributes.parameter_map))
        return AttributeError()
    end

    #Get submodel name and parameter name from parameter map
    (submodel_name, parameter_name) = getfield(multisubmodel_attributes.parameter_map, merged_parameter_name)

    #Extract and return the parameter value from the correct submodel
    return get_parameters(getfield(multisubmodel_attributes.attributes_submodels, submodel_name), parameter_name)
    
end

function ActionModels.get_states(multisubmodel_attributes::MultiSubmodelAttributes, merged_state_name::Symbol)
    #Check if state name is in state map, if not return AttributeError
    if !(merged_state_name in keys(multisubmodel_attributes.state_map))
        return AttributeError()
    end

    #Get submodel name and parameter name from parameter map
    (submodel_name, state_name) = getfield(multisubmodel_attributes.state_map, merged_state_name)

    #Extract and return the parameter value from the correct submodel
    return get_states(getfield(multisubmodel_attributes.attributes_submodels, submodel_name), state_name)
end

#Set specific parameter or state
function ActionModels.set_parameters!(
    multisubmodel_attributes::MultiSubmodelAttributes,
    merged_parameter_name::Symbol,
    parameter_value::T,
) where {T<:Real}

    #Check if parameter name is in parameter map, if not return AttributeError
    if !(merged_parameter_name in keys(multisubmodel_attributes.parameter_map))
        return AttributeError()
    end

    #Get submodel name and parameter name from parameter map
    (submodel_name, parameter_name) = getfield(multisubmodel_attributes.parameter_map, merged_parameter_name)

    #Extract the correct submodel and set the parameter value
    set_parameters!(getfield(multisubmodel_attributes.attributes_submodels, submodel_name), parameter_name, parameter_value)

    return nothing    
end
function ActionModels.set_states!(
    multisubmodel_attributes::MultiSubmodelAttributes,
    merged_state_name::Symbol,
    state_value::T,
) where {T<:Real}

    #Check if state name is in state map, if not return AttributeError
    if !(merged_state_name in keys(multisubmodel_attributes.state_map))
        return AttributeError()
    end

    #Get submodel name and parameter name from parameter map
    (submodel_name, state_name) = getfield(multisubmodel_attributes.state_map, merged_state_name)

    #Extract the correct submodel and set the state value
    set_states!(getfield(multisubmodel_attributes.attributes_submodels, submodel_name), state_name, state_value)

    return nothing
end