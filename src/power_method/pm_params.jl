struct PMParams
    truncp::TruncParams
    itermax::Int64
    eps_converged::Float64
    increase_chi::Bool
    opt_method::String
    normalization::String
    compute_fidelity::Bool

    PMParams(t::TruncParams,nn::Int, epsi::Float64=1e-6, 
        ichi::Bool=false, opt::String="RDM", 
        norm::String="norm", fide::Bool=true) = new(t,nn,epsi,ichi,opt,norm,fide)

end

function PMParams(; truncp=TruncParams(), itermax::Int=200,
    eps_converged::Float64=1e-5, increase_chi::Bool=false, opt_method::String="RTM_LR", normalization::String="norm", compute_fidelity::Bool=true)
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity)
end


function PMParams(p::PMParams; truncp=p.truncp, itermax=p.itermax,
    eps_converged=p.eps_converged, increase_chi=p.increase_chi, 
    opt_method=p.opt_method, normalization=p.normalization, compute_fidelity=p.compute_fidelity)
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity)
end