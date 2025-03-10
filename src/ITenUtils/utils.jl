function dictfromlist(v::Vector)
    dd = Dict()
    for key in v
        dd[key] = []
    end
    return dd
end

""" Updates the first dict with data (with matching keys) from the second """
function mergedicts!(dict_to_update::Dict, new_data::Dict)
    for (key,val) in new_data
        append!(dict_to_update[key], val)
    end
end




""" Takes output of a funcion and saves all elements in a jld2 """
function cpsave(myfunc::Function, cp_name = "checkpoint.jld2")
    #TODO change default filename as date/time ? 
    f = myfunc()
    jldsave(cp_name; f)
    @info "Saving checkpoint to $(cp_name)"
end

function equal_up_to_trailing_zeros(v1::Vector, v2::Vector)
    min_length = min(length(v1), length(v2))
    longer_vector = length(v1) > length(v2) ? v1 : v2
    
    # Compare elements up to the length of the shorter vector
    if !all(v1[1:min_length] .== v2[1:min_length])
        return false
    end
    
    # Check if remaining elements in the longer vector are all zeros
    return all(x -> abs(x) < 1e-15, longer_vector[min_length+1:end])
end