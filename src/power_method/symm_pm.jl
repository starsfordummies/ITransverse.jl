"""
Power method for *symmetric* case: takes as input a single MPS psi and an MPO O,
    applies the mpo and optimizes the overlap <psiO*|Opsi> 
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams)

    (; itermax, truncp, normalization, compute_fidelity) = pm_params

    stepper, info_iterations, maxdims = init_pm(pm_params)
    
    # normalize initial boundary for stability
    psi_work = normalize(in_mps)

    p = Progress(itermax; desc="[Symmetric PM|$(truncp.alg)] L=$(length(in_mps)), cutoff=$(truncp.cutoff), χmax=$(maxdims[end]), normalize=$(normalization))", showspeed=true) 


    for jj = 1:itermax

        if compute_fidelity
            psi_prev = copy(psi_work)
        end

        psi_work, sv = tapply(in_mpo, psi_work; maxdim=maxdims[jj], truncp...)

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
    #replace_linkinds!(psi_work, "Link,rotl=")
    return psi_work, info_iterations

end
