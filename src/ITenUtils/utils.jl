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
