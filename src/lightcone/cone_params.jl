struct ConeParams{T<:Number, TP}
    truncp::TP
    opt_method::Symbol
    optimize_op::Vector{T}
    vwidth::Int 
end

""" Defaults for everything except truncp """
function ConeParams(; truncp,
    opt_method::Symbol = :sym,  # either R or LR
    optimize_op::Vector{<:Number} = [1,0,0,1],
    vwidth::Int=1) 

    return ConeParams(truncp, opt_method, optimize_op, vwidth)

end