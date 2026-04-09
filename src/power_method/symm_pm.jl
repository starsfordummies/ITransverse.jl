"""
Power method for *symmetric* case: takes as input a single MPS psi and an MPO O,
    applies the mpo and optimizes the overlap <psiO*|Opsi> 
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams; normalize_psi0::Bool=false)

    (; itermax, truncp, cutoffs, maxdims, normalization, compute_fidelity) = pm_params

    stepper, info_iterations = init_pm(pm_params)
    
    # normalize initial boundary for stability
    psi_work = normalize_psi0 ? normalize(in_mps) : in_mps

    use_eig = get(truncp, :use_eig, false)
    use_eig_string = use_eig ? "EIG" : "SVD"

    pm_info_string = "[Symmetric PM|$(truncp.alg)|$(use_eig_string)] L=$(length(in_mps)), cutoff=$(cutoffs[end]), χmax=$(maxdims[end]), normalize=$(normalization))"

    p = Progress(itermax; desc=pm_info_string, showspeed=true) 

    eltype_S = use_eig ? ComplexF64 : Float64 

    sv_prev = zeros(eltype_S, 2,2)

    for jj = 1:itermax

        maxdim = get(maxdims, jj, maxdims[end])
        cutoff = get(cutoffs, jj, cutoffs[end])
        truncp = merge(pm_params.truncp, (;cutoff, maxdim))


        if compute_fidelity
            psi_prev = copy(psi_work)
        end

        psi_work, sv = tapply(in_mpo, psi_work; truncp...)

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


        stop, reason = pm_itercheck!(stepper, info_iterations, psi_work, sv_prev, sv)
        sv_prev = sv 

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
