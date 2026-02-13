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

    (; itermax, opt_method, truncp, normalization, compute_fidelity) = pm_params

    stepper, info_iterations, maxdims = init_pm(pm_params)
    
    # normalize initial boundary for stability
    psi_work = normalize(in_mps)

    p = Progress(itermax; desc="[Symmetric PM|$(opt_method)] L=$(length(in_mps)), cutoff=$(truncp.cutoff), χmax=$(maxdims[end]), normalize=$(normalization))", showspeed=true) 


    for jj = 1:itermax

        if opt_method == "RTMRDM" && jj == div(itermax,2)
            #increase_chi = false
            max_chi = maxlinkdim(psi_work)+4 # give it some room for adjustment
            opt_method = "RDM"
            @info "$(jj) - Changing method to RDM "
        end

        if compute_fidelity
            psi_prev = copy(psi_work)
        end

        psi_work, sv = if opt_method == "RDM"
            tapply(Algorithm("densitymatrix"), in_mpo, psi_work; cutoff=truncp.cutoff, maxdim=maxdims[jj])
        elseif opt_method == "RTM" || opt_method == "RTMRDM"
            tapply(Algorithm("RTMsym"), in_mpo, psi_work; cutoff=truncp.cutoff, maxdim=maxdims[jj], method="SVD", fast)
        elseif opt_method == "RTM_EIG" # this can be less accurate
            psi_work, sv = tapply(Algorithm("RTMsym"), in_mpo, psi_work; cutoff=truncp.cutoff, maxdim=maxdims[jj], method="EIG", fast)
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


        stop, reason = pm_itercheck!(stepper, info_iterations, psi_work, sv)

        # should we stop?
        if stop
            if reason == :converged
                @info "PM Converged after $jj steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(psi_work))"
            elseif reason == :stuck
                @warn "PM Stuck after $(stepper.iters_without_improvement)/$(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(psi_work))"
            end
          
            break
        end

        if jj == itermax
            @warn "PM **not** converged after $(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(psi_work))"
        end


        next!(p; showvalues = [(:Info,"[$(jj)]  chi=$(maxlinkdim(psi_work)) | ds2=$(last(info_iterations[:ds])) | <R|Rprev> = $(fidelity)" )])

            

    end

    # nicer link labels at the end
    replace_linkinds!(psi_work, "Link,rotl=")
    return psi_work, info_iterations

end
