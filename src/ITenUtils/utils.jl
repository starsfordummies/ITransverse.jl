function dictfromlist(v::Vector{T}, ValueType=Any) where T
    dd = Dict{T, Vector{ValueType}}()
    for key in v
        dd[key] = ValueType[]  
    end
    return dd
end

function smart_append!(A, B)
    if A isa AbstractArray && B isa AbstractArray
        append!(A, B)
    elseif A isa AbstractArray && B isa Number
        push!(A, B)
    elseif A isa Number && B isa Number
        return [A, B]
    elseif A isa Number && B isa AbstractArray
        return vcat([A], B)
    else
        throw(ArgumentError("Unsupported argument types"))
    end
    return A
end

""" Updates the first dict with data (with matching keys) from the second """
function mergedicts!(dict_to_update::Dict, new_data::Dict)
    for (key,val) in new_data
        if !haskey(dict_to_update, key)
            @warn "key $(key) not present in the dict to update, creating an empty one"
            dict_to_update[key] = []  # Initialize with an empty array if the key doesn't exist
        end
        
        smart_append!(dict_to_update[key], val)
    
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


function convert_keys_to_symbols_recursive(dict::Dict)
    return Dict(Symbol(k) => (v isa Dict ? convert_keys_to_symbols_recursive(v) : v) for (k, v) in dict)
end


function beta_lims(Ntot::Int, nbeta::Int, init_beta_only::Bool)

    b1 = if init_beta_only 
        nbeta
    else
        if iseven(nbeta)
            div(nbeta,2)
        elseif nbeta == 1 
            nbeta
        else
            error("even nbeta")
        end  
    end

    b2 = if init_beta_only 
        Ntot 
    else
        Ntot - div(nbeta,2) 
    end

    return b1,b2
end