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
        if !haskey(dict_to_update, key)
            @warn "key $(key) not present in the dict to update, creating an empty one"
            dict_to_update[key] = []  # Initialize with an empty array if the key doesn't exist
        end
        if dict_to_update[key] isa Vector && val isa Vector
            append!(dict_to_update[key], val)
        end
    end
end

function mergedicts(dict1::Dict, dict2::Dict)
    result = deepcopy(dict1)
    for (key, val) in dict2
        if haskey(result, key)
            if val isa Dict && result[key] isa Dict
                # Recursively merge nested dictionaries
                result[key] = mergedicts(result[key], val)
            elseif val isa AbstractVector && result[key] isa AbstractVector
                # Create a new vector by concatenating
                result[key] = vcat(result[key], val)
            else
                # Overwrite the value if types differ or not handled above
                @warn "Overwriting $(key) ?!" 
                result[key] = val
            end
        else
            # Add new key-value pair if key doesn't exist
            result[key] = val
        end
    end
    return result
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

function convert_keys_to_symbols_recursive(dict::Dict)
    return Dict(Symbol(k) => (v isa Dict ? convert_keys_to_symbols_recursive(v) : v) for (k, v) in dict)
end