"""
Power method for *symmetric* case: takes as input a single MPS |L>,
    applies the `in_mpo` O and optimizes the overlap (LO|OL) , where LO is *not* conjugated.
    Works eg. for unfolded Loschmidt echo type setups, where `in_mpo` is the transfer matrix 
    and truncating over (LO|OL) does not give trivial results.
    We can nevertheless specify in `pm_params` the `opt_method = RDM` if we want to use the usual truncation
    based on the reduced density matrix (which boils down to SVD of the Left vector)
    For the RTM truncation, since the problem is symmetric, we can choose to truncate over the SV of 
    the symmetric environments (`opt_method=RTM`), which are computed in a Autonne-Takagi form, 
    or compute the symmetric eigenvalue problem of the RTM   (`opt_method=RTM_EIG`)
    and truncate over the (complex, so be mindful..) eigenvalues of the RTM. 
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams; fast::Bool=false)

    (; itermax, eps_converged, opt_method, truncp, increase_chi, normalization, compute_fidelity) = pm_params
    (; cutoff, maxbondim) = truncp

    # Normalize eps_converged by system size or larger chains will never converge as good...
    eps_converged = eps_converged * length(in_mps)
    stopper = PMstopper(pm_params; eps_converged)
  
    # normalize initial boundary for stability
    psi_work = normalize(in_mps)

    ds2s = [] 
    ds2 = 0. 
    sprevs = ones(length(in_mps)-1, maxlinkdim(in_mps)*maxlinkdim(in_mpo))

    p = Progress(itermax; desc="[Symmetric PM|$(opt_method)] L=$(length(in_mps)), cutoff=$(cutoff), χmax=$(maxbondim), normalize=$(normalization))", showspeed=true) 

    max_chi = maxbondim
    maxbondim = 20 

    for jj = 1:itermax

        if opt_method == "RTMRDM" && jj == div(itermax,2)
            #increase_chi = false
            max_chi = maxlinkdim(psi_work)+4 # give it some room for adjustment
            opt_method = "RDM"
            @info "$(jj) - Changing method to RDM "
        end

        if compute_fidelity
            psi_prev = copy(psi_work)
            prev_norm = norm(psi_prev)
        end

        if increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim, max_chi])
        else
            maxbondim = max_chi
        end

        psi_work, sv = if opt_method == "RDM"
            tapply(Algorithm("densitymatrix"), in_mpo, psi_work; cutoff=cutoff, maxdim=maxbondim)
        elseif opt_method == "RTM" || "RTMRDM"
            tapply(Algorithm("RTMsym"), in_mpo, psi_work; cutoff=cutoff, maxdim=maxbondim, method="SVD", fast)
        elseif opt_method == "RTM_EIG" # this can be less accurate
            tapply(Algorithm("RTMsym"), in_mpo, psi_work; cutoff=cutoff, maxdim=maxbondim, method="EIG", fast)
        else
            error("Specify a valid opt_method: RDM|RTM|...")
        end


        if normalization == "norm"
            #orthogonalize!(psi_ortho,1)
            normalize!(psi_work)
        elseif normalization == "overlap"
            # normalize so that <L|R> = 1 
            overlap = overlap_noconj(psi_work,psi_work)
            psi_work = psi_work / sqrt(overlap)
        end # otherwise we do nothing - norm can blow up! 

        fidelity = if compute_fidelity 
           abs( log(abs(inner(psi_work, psi_prev))) - log(norm(psi_work)) - log(norm(psi_prev)) ) / length(psi_work)
        else 
            NaN
        end

        ds2 = max_diff(sprevs, sv) 
        @show jj, ds2

        push!(ds2s, [ds2, fidelity])

        sprevs = sv

        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), <R|Rprev> = $(fidelity), chi=$(maxlinkdim(psi_work))" )])

        if jj > 100
            stop, reason = should_stop_ds2!(stopper,ds2)

            # should we stop?
            if stop
                if reason == :converged
                    @info "Converged after $jj steps (ds2=$(ds2))"
                elseif reason == :stuck
                    @warn "Iteration stuck after $jj steps (ds2=$(ds2)); stopping."
                end
                break
            end

        end

    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(psi_work))")

    # nicer link labels
    replace_linkinds!(psi_work, "Link,rotl=")
    return psi_work, ds2s

end
