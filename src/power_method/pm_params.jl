#= struct PMParams
    truncp::TruncParams
    itermin::Int
    itermax::Int
    eps_converged::Float64
    increase_chi::Bool
    opt_method::String
    normalization::String
    compute_fidelity::Bool
    stuck_after::Int
    quiet::Bool

    PMParams(t::TruncParams, 
        itermax::Int=1000, 
        itermin::Int=1,
        epsi::Float64=1e-6, 
        ichi::Bool=false, 
        opt::String="RDM", 
        norm::String="norm", 
        fide::Bool=true,
        nstuck::Int=itermax,
        quiet::Bool=false) = new(t,itermin,itermax,epsi,ichi,opt,norm,fide,nstuck,quiet)

end

function PMParams(; 
    truncp=TruncParams(), 
    itermin::Int=1,
    itermax::Int=200,
    eps_converged::Float64=1e-5, 
    increase_chi::Bool=false,
    opt_method::String="RTM_LR",
    normalization::String="norm", 
    compute_fidelity::Bool=true,
    stuck_after::Int=itermax)
    return PMParams(truncp, itermin, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity, stuck_after)
end


function PMParams(p::PMParams; truncp=p.truncp, itermin=p.itermin, itermax=p.itermax,
    eps_converged=p.eps_converged, increase_chi=p.increase_chi, 
    opt_method=p.opt_method, normalization=p.normalization, compute_fidelity=p.compute_fidelity,
    stuck_after=p.stuck_after)
    return PMParams(truncp, itermin, itermax, eps_converged, increase_chi, opt_method, normalization, compute_fidelity, stuck_after)
end

=#

Base.@kwdef mutable struct PMParams
    truncp::TruncParams = TruncParams()
    itermin::Int = 20
    itermax::Int = 600
    eps_converged::Float64 = 1e-8
    increase_chi::Bool = true
    opt_method::String = "RDM"
    normalization::String = "norm"
    compute_fidelity::Bool = true
    stuck_after::Int = itermax
    quiet::Bool = false
end

function make_chi_table(increase_chi::Bool, maxdim::Int, itermax::Int, 
                          start_chi::Int=20, step::Int=2)
    if !increase_chi
        return fill(maxdim, itermax)
    end
    
    schedule = Vector{Int}(undef, itermax)
    for i in 1:itermax
        schedule[i] = min(start_chi + (i-1)*step, maxdim)
    end
    return schedule
end


function make_chi_table(pm_params::PMParams; start_chi=20, step::Int=2)
    (; increase_chi, truncp, itermax) = pm_params
    make_chi_table(increase_chi, truncp.maxdim, itermax, start_chi, step)

end


# Iteration step checker

Base.@kwdef mutable struct PMstep{T <: Number}
    icurr::Int = 0
    best_ds::Float64 = Inf
    prev_s::Matrix{T}
    iters_without_improvement::Int = 0

    # Convergence criteria 
    eps_converged::Float64
    eps_max::Float64 = 0.05
    stuck_after::Int = 100
    itermin::Int = 20

end

function PMstep(pm_params::PMParams; 
    eps_converged=pm_params.eps_converged, 
    stuck_after::Int=pm_params.stuck_after,
    itermin::Int=pm_params.itermin)

    elt = pm_params.opt_method == "RTM_EIG" ? ComplexF64 : Float64
    prev_s = zeros(elt, 2,2)

    PMstep(;prev_s, eps_converged, stuck_after, itermin)
end


function init_pm(pm::PMParams)

    # Iteration History 
    stepper = PMstep(pm)
    info_iterations = Dict(
        :ds => Float64[],
        :fidelity => Float64[],
        :chi => Int[]
    )
    chi_table = make_chi_table(pm)

    return stepper, info_iterations, chi_table
end


function pm_itercheck!(
    stepper::PMstep,
    info_iterations::Dict,
    psi::MPS,
    Smat::Matrix)
    

    ds = stepper.icurr == 0 ? Inf : max_diff(stepper.prev_s, Smat) 
    stepper.prev_s = Smat

    chi = maxlinkdim(psi)
    
    # Append to history
    push!(info_iterations[:ds], ds)
    push!(info_iterations[:chi], chi)
    
    stepper.icurr += 1 

    if stepper.icurr < stepper.itermin
        return false, :keep_going
    else # Check for convergence
        # Absolute convergence
        if ds < stepper.eps_converged
            return true, :converged
        elseif ds > stepper.eps_max
            return false, :keep_going
        else
            if ds < stepper.best_ds
                stepper.best_ds = ds
                stepper.iters_without_improvement = 0
            else
                stepper.iters_without_improvement += 1
            end

            if stepper.iters_without_improvement > stepper.stuck_after
                return true, :stuck 
            else
                return false, :keep_going
            end
        end
    end


end

