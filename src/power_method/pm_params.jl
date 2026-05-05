Base.@kwdef mutable struct PMParams{TP,TChis,TCutoffs}
    truncp::TP = (cutoff=1e-12, maxdim=256, direction=:right, alg="densitymatrix")
    opt_method::Symbol = :sym  # either R or LR
    itermin::Int = 20
    itermax::Int = 600
    eps_converged::Float64 = 1e-6
    maxdims::TChis = 2:2:truncp.maxdim
    cutoffs::TCutoffs = [truncp.cutoff]
    normalization::String = "norm"
    compute_fidelity::Bool = true
    stuck_after::Int = itermax
    quiet::Bool = false
end

function Base.show(io::IO, p::PMParams)
    println(io, "PMParams:")
    for f in fieldnames(PMParams)
        println(io, "  $(rpad(f, 16)) = $(getfield(p, f))")
    end
end




# Iteration step checker

Base.@kwdef mutable struct PMstep
    icurr::Int = 0
    best_ds::Float64 = Inf
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

    PMstep(;eps_converged, stuck_after, itermin)
end


""" Returns PMstep stepper and Dict info_iterations """
function init_pm(pm::PMParams)

    # Iteration History 
    stepper = PMstep(pm)
    info_iterations = Dict(
        :ds => Float64[],
        :fidelity => Float64[],
        :chi => Int[]
    )

    return stepper, info_iterations
end


function pm_itercheck!(
    stepper::PMstep,
    info_iterations::Dict,
    psi::MPS,
    prev_s::Matrix,
    Smat::Matrix)
    
    #@show sum(Smat[10,:])

    ds = stepper.icurr == 0 ? Inf : max_diff(prev_s, Smat) 

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

