

function mergedicts!(dict1::Dict, dict2::Dict)
    for key in keys(dict2)
        append!(dict1[key], dict2[key])
    end
end




""" Takes output of a funcion and saves all elements in a jld2 """
function cpsave(myfunc::Function, cp_name = "checkpoint.jld2")
    #TODO change default filename as date/time ? 
    f = myfunc()
    jldsave(cp_name; f)
    @info "Saving checkpoint to $(cp_name)"
end
