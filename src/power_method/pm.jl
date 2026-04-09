"""
Power method developed for the folding algorithm, takes as input TWO MPOs, 
    one meant to be with an additional operator (X), the other likely with an identity (1)
"""
function powermethod_op(in_mps::MPS; mpo_id::MPO, mpo_op::MPO, pm_params::PMParams)

    (; opt_method, itermax, maxdims, cutoffs, normalization, compute_fidelity) = pm_params

    stepper, info_iterations = init_pm(pm_params)

    ll = copy(in_mps)
    rr = copy(in_mps)

    sv_prev = zeros(Float64, 2,2)
    pm_info_string = "[PM(OP)|$(pm_params.truncp.alg)|$(opt_method)] L=$(length(in_mps)), cutoff=$(last(cutoffs)), maxdim=$(last(maxdims)), normalize=$(normalization))"


    p = Progress(itermax; desc=pm_info_string, showspeed=true) 

    for jj = 1:itermax  

        maxdim = get(maxdims, jj, maxdims[end])
        cutoff = get(cutoffs, jj, cutoffs[end])
        truncp = merge(pm_params.truncp, (;cutoff, maxdim))

        rr_prev = compute_fidelity ? copy(rr) : nothing


        ll, rr, SVs = if opt_method == :sym

            rright, SVs = if truncp.alg == "densitymatrix" || truncp.alg == "naive"
                tapply(mpo_id, rr; truncp...)
            else
                _, rright, SVs = tlrapply(ll, mpo_op, mpo_id, rr; truncp...)
                rright, SVs
            end

            sim(linkinds, rright), rright, SVs

        else # not sym  

            #@info "non-symmetric"

            ll, _, _   = tlrapply(ll, mpo_id, mpo_op, rr; truncp...)
            _, rr, SVs = tlrapply(ll, mpo_op, mpo_id, rr; truncp...)

            ll, rr, SVs

        end # TODO: option for applying more than one col? 


        # Normalize after each step 
        if normalization == "norm"
            #ll = orthogonalize!(ll,1)
            #rr = orthogonalize!(rr,1)

            ll = normalize(ll)
            rr = normalize(rr)
        elseif normalization == "overlap"
            normalize_for_overlap!(ll,rr)
        end # no normalization - could blow up 


        logfidelityRRnew = compute_fidelity ? logfidelity(rr_prev,rr) : NaN


        stop, reason = pm_itercheck!(stepper, info_iterations, rr, sv_prev, SVs)
        sv_prev = SVs

        if stop
            if reason == :converged
                @info "PM Converged after $jj steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(rr))"
            elseif reason == :stuck
                @warn "PM Stuck after $(stepper.iters_without_improvement)/$(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(rr))"
            end
        
            break
        end

        if jj == itermax
            @warn "PM **not** converged after $(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(maxlinkdim(rr))"
        end

        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(last(info_iterations[:ds])), logfidelity(<R|Rnew>)=$(logfidelityRRnew)" )])

    end

    return ll, rr, info_iterations

end




"""
Power method for a non-symmetric case (no extra operator)
Starts with |ψ0>, 
builds <L| = <ψ0|(in_mpo_L)^N->inf
 (MPO inds are swapped so to apply it to the left)
and  |R> = (in_mpo_R)^N|ψ0>

"""
function powermethod_lr(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::PMParams)

    (; opt_method, itermax, cutoffs, maxdims, truncp, normalization, compute_fidelity) = pm_params

    stepper, info_iterations = init_pm(pm_params)

    ll, rr, svs = tlrapply(in_mps, in_mpo_L, in_mpo_R, in_mps; truncp...)
   
    llprev = copy(ll)

    sv_prev = zeros(Float64, 2,2)
    pm_info_string = "[PM LR|$(pm_params.truncp.alg)|$(opt_method)] L=$(length(in_mps)), cutoff=$(last(cutoffs)), maxdim=$(last(maxdims)), normalize=$(normalization))"

    p = Progress(itermax; desc=pm_info_string, showspeed=true) 
    
    for jj = 1:itermax

        ll, rr, svs = tlrapply(ll, in_mpo_L, in_mpo_R, rr; truncp...)

        if normalization == "norm"
            ll = normalize(ll)
            rr = normalize(rr)
        elseif normalization == "overlap"
            ov = overlap_noconj(ll,rr)
            ll = ll/sqrt(ov)
            rr = rr/sqrt(ov)
        end  # otherwise do nothing, norms can blow up 

        fidelity_step = if compute_fidelity
            lfid = logfidelity(ll, llprev)
            llprev = copy(ll)
            lfid 
        else
            NaN
        end


        chi_max = max(maxlinkdim(ll),maxlinkdim(rr))
        

        stop, reason = pm_itercheck!(stepper, info_iterations, rr, sv_prev, svs)
        sv_prev = svs

        # should we stop?
        if stop
            if reason == :converged
                @info "PM Converged after $jj steps | ds=$(last(info_iterations[:ds])) | chi=$(chi_max))"
            elseif reason == :stuck
                @warn "PM Stuck after $(stepper.iters_without_improvement)/$(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(chi_max))"
            end
        
            break
        end

        if jj == itermax
            @warn "PM **not** converged after $(jj) steps | ds=$(last(info_iterations[:ds])) | chi=$(chi_max))"
        end

        next!(p; showvalues = [(:Info,"[$(jj)]  chi=$(chi_max) | ds=$(last(info_iterations[:ds])) | <R|Rprev> = $(fidelity_step)" )])

    end

    return ll, rr, info_iterations

end
