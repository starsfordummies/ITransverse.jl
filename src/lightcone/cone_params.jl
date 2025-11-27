struct ConeParams{T<:Number}
    truncp::TruncParams
    opt_method::String
    optimize_op::Vector{T}
    which_evs::Vector{String}
    which_ents::Vector{String}
    checkpoints::Tuple{Int}
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
        (50:checkpoints:10000,)         
    elseif isa(checkpoints, Tuple)
        checkpoints                              # tuple → keep as is
    elseif isa(checkpoints, AbstractVector)
        tuple(checkpoints...) 
    elseif isa(checkpoints, AbstractRange)
        tuple(checkpoints...)                    # range → tuple
    else
        throw(ArgumentError("Unsupported input type $(typeof(checkpoints))"))
    end


    return ConeParams(truncp, opt_method, optimize_op, which_evs, which_ents, checkpoints, vwidth)

end