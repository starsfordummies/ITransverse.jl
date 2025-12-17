struct ConeParams{T<:Number}
    truncp::TruncParams
    opt_method::String
    optimize_op::Vector{T}
    which_evs::Vector{String}
    which_ents::Vector{String}
    checkpoints::Vector{Int}
    vwidth::Int 
end

""" Defaults for everything except truncp """
function ConeParams(; truncp::TruncParams,
    opt_method::String = "RDM",
    optimize_op::Vector{<:Number} = [1,0,0,1],
    which_evs::Vector{String} = ["X"],
    which_ents::Vector{String}= [""],
    checkpoints,
    vwidth::Int=1) 

    checkpoints = if isa(checkpoints, Integer)
        if checkpoints > 0 
            collect(50:checkpoints:10000)         
        else
            Int[]
        end
    elseif isa(checkpoints, Tuple{Int})
        checkpoints                              # tuple → keep as is
    elseif isa(checkpoints, AbstractVector{Int})
        collect(checkpoints) 
    elseif isa(checkpoints, AbstractRange{Int})
        collect(checkpoints)                    # range → tuple
    else
        throw(ArgumentError("Unsupported input type $(typeof(checkpoints))"))
    end


    return ConeParams(truncp, opt_method, optimize_op, which_evs, which_ents, checkpoints, vwidth)

end