struct PMParams
    truncp::TruncParams
    itermax::Int64
    eps_converged::Float64
    increase_chi::Bool
    opt_method::String
end


function PMParams(; truncp=TruncParams(), itermax::Int=200,
    eps_converged::Float64=1e-5, increase_chi::Bool=false, opt_method::String="RTM_LR")
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method)
end


function PMParams(p::PMParams; truncp=p.truncp, itermax=p.itermax,
    eps_converged=p.eps_converged, increase_chi=p.increase_chi, opt_method=p.opt_method)
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method)
end