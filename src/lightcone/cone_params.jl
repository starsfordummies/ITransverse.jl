struct ConeParams{T<:Number}
    truncp::TruncParams
    opt_method::String
    optimize_op::Vector{T}
    which_evs::Vector{String}
    which_ents::Vector{String}
    checkpoint::Int
end

ConeParams(; truncp::TruncParams,
    opt_method::String = "RTM_LR",
    optimize_op::Vector{<:Number} = [1,0,0,1],
    which_evs::Vector{String} = ["X"],
    which_ents::Vector{String}= [""],
    checkpoint::Int=20) = ConeParams(truncp, opt_method, optimize_op, which_evs, which_ents, checkpoint)
