struct ConeParams{T<:Number}
    truncp::TruncParams
    opt_method::String
    optimize_op::Vector{T}
    vwidth::Int 
end

""" Defaults for everything except truncp """
function ConeParams(; truncp::TruncParams,
    opt_method::String = "RDM",
    optimize_op::Vector{<:Number} = [1,0,0,1],
    vwidth::Int=1) 

    return ConeParams(truncp, opt_method, optimize_op, vwidth)

end