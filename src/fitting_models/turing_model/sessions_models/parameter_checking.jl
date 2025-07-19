### Outer-layer sessions Turing submodel for parameter rejections ###
#Just calls the speified sessions_model with a NoParameterChecking marker
@model function sessions_model(
    check_rejections_marker::ParameterChecking,
    pos_args...;
    kwargs...,
)
    #Run the normal session model
    try
        i ~ to_submodel(
            sessions_model(
                NoParameterChecking(),
                #Unpack arguments  
                pos_args...;
                kwargs...,
            ),
            false, #Do not add a prefix
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

