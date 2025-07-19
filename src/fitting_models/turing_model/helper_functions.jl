#######################################################################
### FUNCTION FOR RESHAPING A VECTOR OF TUPLES TO A TUPLE OF VECTORS ###
#######################################################################
function evert(v::Vector{<:NTuple{N,Any}}) where {N}
    n = length(v)

    # Create result vectors with the correct types
    sample = first(v)
    result = ntuple(i -> Vector{typeof(sample[i])}(undef, n), N)

    # Fill each vector with values from the corresponding tuple position
    @inbounds for i = 1:n
        t = v[i]
        for j = 1:N
            result[j][i] = t[j]
        end
    end

    return result
end

### DISPATCH FOR THE CASE WHEN IT IS REALS OR DISTS INSTEAD OF TUPLE ###
function evert(v::Vector{T}) where {T}
    return (v,)
end