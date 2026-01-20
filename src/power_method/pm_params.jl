struct PMParams
    truncp::TruncParams
    itermax::Int
    eps_converged::Float64
    increase_chi::Bool
    opt_method::String
    normalization::String
    compute_fidelity::Bool
    stuck_after::Int
    quiet::Bool

    PMParams(t::TruncParams, 
        nn::Int, 
        epsi::Float64=1e-6, 
        ichi::Bool=false, 
        opt::String="RDM", 
        norm::String="norm", 
        fide::Bool=true,
        nstuck::Int=nn,
        quiet::Bool=false) = new(t,nn,epsi,ichi,opt,norm,fide,nstuck,quiet)

end

function PMParams(; 
    truncp=TruncParams(), 
    itermax::Int=200,
    eps_converged::Float64=1e-5, 
    increase_chi::Bool=false,
    opt_method::String="RTM_LR",
    normalization::String="norm", 
    compute_fidelity::Bool=true,
    stuck_after::Int=itermax)
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity, stuck_after)
end


function PMParams(p::PMParams; truncp=p.truncp, itermax=p.itermax,
    eps_converged=p.eps_converged, increase_chi=p.increase_chi, 
    opt_method=p.opt_method, normalization=p.normalization, compute_fidelity=p.compute_fidelity,
    stuck_after=p.stuck_after)
    return PMParams(truncp, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity, stuck_after)
end


mutable struct PMstopper
    best_ds2::Float64
    iters_without_improvement::Int
    eps_converged::Float64
    eps_max::Float64
    stuck_after::Int
end

function PMstopper(pmp::PMParams; init_ds2 = Inf, eps_converged=pmp.eps_converged, eps_max::Float64=1e-2, stuck_after::Int=pmp.stuck_after)
    PMstopper(init_ds2, 0, eps_converged, eps_max, stuck_after)
end

function should_stop_ds2!(
    stopper::PMstopper,
    ds2::Float64
)
    # Absolute convergence
    if ds2 < stopper.eps_converged
        return true, :converged
    elseif ds2 > stopper.eps_max
        return false, :keep_going
    else
        if ds2 < stopper.best_ds2
            stopper.best_ds2 = ds2
            stopper.iters_without_improvement = 0
        else
            stopper.iters_without_improvement = stopper.iters_without_improvement + 1
        end

        if stopper.iters_without_improvement > stopper.stuck_after
            return true, :stuck 
        else
            return false, :keep_going
        end
    end

end

