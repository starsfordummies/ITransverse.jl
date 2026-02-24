"""
Power method developed for the folding algorithm, takes as input TWO MPOs, 
    one meant to be with an additional operator (X), the other likely with an identity (1)

Neither of the input MPOs needs to have the L-R indices swapped, we do it in here already.

Depending on pm_params.opt_method, the update can work as follows

- "RTM_LR": At each iteration we make *two* updates, first compute <L1|OR> and updates <Lnew|, 
    then computes <LO|1R> and computes |Rnew>.

- "RTM_R": for a *symmetric* tMPO we should be able to just update one of the two (say |R>), and the
 corresponding should just be the transpose. Note that this is *not* the same as performing a symmetric update
 of the form <R|R>, here we still update the overlap <LO|1R>

- "RDM": the common truncation using temporal entanglement, ie. over the RDM (not RTM) of |R>. 
   In practice this is done with the usual SVD truncations of R. In this case, the `in_mpo_O` input is unused.


At each step of the PM we want to normalize back the tMPS, or in the long run we lose precision. 
The most consistent way to do it is probably to enforce that the overlap <L|R> = 1 (pm_params.normalize="overlap"), but in practice normalizing individually
<L|L> = <R|R> = 1 (pm_params.normalize="norm") seems to work as well. 

Truncation params are in `pm_params.truncp`

We return the (hopefully converged) |R> and <L| tMPS, 
and some check quantities along the PM iterations in a dict info_iterations
the ΔS^2 for the SV calculated,
the norms <R|Rnew>,
the difference <L|R> - <Lnew|Rnew> 

We could also return the (optimized) LO and OR calculated, but if we converge those should be easily calculable afterwards
by applying the relevant MPO to the resulting leading eigenvectors.

"""



function powermethod_op(in_mps::MPS, in_mpo_1::MPO, in_mpo_O::MPO, pm_params::PMParams)

    (; opt_method, itermax, truncp, normalization, compute_fidelity) = pm_params
    cutoff=truncp.cutoff

    stepper, info_iterations, maxdims = init_pm(pm_params)

    ll = copy(in_mps)
    rr = copy(in_mps)

    p = Progress(itermax; desc="[PM|$(opt_method)] L=$(length(ll)), cutoff=$(cutoff), maxdim=$(last(maxdims)), normalize=$(normalization)", showspeed=true) 

    for jj = 1:itermax  

        rr_prev = compute_fidelity ? copy(rr) : nothing

        if opt_method == "RTM_LR"
    
            # optimize <LO|1R> -> new |R> 
            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL; cutoff, maxdim=maxdims[jj], direction=truncp.direction)

            # optimize <L1|OR> -> new <L|  
            #TODO: we could be using either the new rr here or the previous rr (in that case should define rr_work = rr before)
            OpsiR = applyn(in_mpo_O, rr)
            OpsiL = applyns(in_mpo_1, ll)  

            _, ll, _ = truncate_sweep(OpsiR, OpsiL; cutoff, maxdim=maxdims[jj], direction=truncp.direction)

        elseif opt_method == "RTM_R"

            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL; cutoff, maxdim=maxdims[jj], direction=truncp.direction)
            ll = rr

        elseif opt_method == "RTM_R_twolayers" # expensive

            OpsiR = applyn(in_mpo_1, rr)
            OpsiR = applyn(in_mpo_1, OpsiR)

            OpsiL = applyns(in_mpo_1, rr)  
            OpsiL = applyns(in_mpo_O, OpsiL)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL; cutoff, maxdim=maxdims[jj], direction=truncp.direction)
            ll = rr

        elseif opt_method == "RDM"
        
            ll, _ =  tapplys(in_mpo_1, ll; alg="densitymatrix", cutoff, maxdim=maxdims[jj])
            rr, SVs = tapply(in_mpo_1, rr; alg="densitymatrix", cutoff, maxdim=maxdims[jj])

        elseif opt_method == "RDM_R"
   
            rr, SVs = tapply(in_mpo_1, rr; cutoff, maxdim=maxdims[jj])
            ll = rr

        else
            @error "Wrong optimization method: $opt_method"
        end


        # Normalize after each step 
        if normalization == "norm"
            ll = orthogonalize!(ll,1)
            rr = orthogonalize!(rr,1)

            ll = normalize(ll)
            rr = normalize(rr)
        elseif normalization == "overlap"
            normalize_for_overlap!(ll,rr)
        end # no normalization - could blow up 


        logfidelityRRnew = compute_fidelity ? logfidelity(rr_prev,rr) : NaN


        chimax = max(maxlinkdim(ll),maxlinkdim(rr))
        push!(info_iterations[:chi], chimax)
     
       stop, reason = pm_itercheck!(stepper, info_iterations, rr, SVs)

        # should we stop?
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



        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(last(info_iterations[:ds]))), logfidelity(<R|Rnew>)=$(logfidelityRRnew)" )])

    end

    return ll, rr, info_iterations

end




"""
Power method *without* operator 

    Starts with |in_mps>, builds <L| = <in_mps|in_mpo_L and |R> = in_mpo_R|in_mps>
    and optimizes them iteratively. 
    Note that at each step, there is a single optimization which returns a new set Lnew, Rnew
    which are used as starting point for the next step. 

    So this is only if we *don't* put in extra operators and the like. For that case, use `powermethod`

Truncation params are in pm_params
"""
function powermethod_both(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::PMParams)

    (; opt_method, itermax, truncp, normalization, compute_fidelity) = pm_params

    stepper, info_iterations, maxdims = init_pm(pm_params)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(truncp.cutoff), χmax=$(last(maxdims)), normalize=$(normalization)", showspeed=true) 

    for jj = 1:itermax

        llprev = deepcopy(ll)

        ll, rr, svs = if opt_method == "RDM"
            ll, _  = tapplys(in_mpo_L, ll; cutoff=truncp.cutoff, maxdim=maxdims[jj])
            rr, svs = tapply(in_mpo_R, rr; cutoff=truncp.cutoff, maxdim=maxdims[jj])

            ll,rr,svs
        else # RTM

            OpsiL = applyns(in_mpo_L, ll)
            OpsiR = applyn(in_mpo_R, rr)

            truncate_sweep(OpsiL, OpsiR, truncp)

        end

        if normalization == "norm"
            ll = normalize(ll)
            rr = normalize(rr)
        elseif normalization == "overlap"
            normalize_for_overlap!(ll,rr)
        end  # otherwise do nothing, norms can blow up 

        fidelity_step = compute_fidelity ? logfidelity(ll, llprev) : NaN

        next!(p; showvalues = [(:Info,"[$(jj)]  chi=$(maxlinkdim(rr)) | ds=$(last(info_iterations[:ds])) | <R|Rprev> = $(fidelity_step)" )])
        stop, reason = pm_itercheck!(stepper, info_iterations, rr, svs)

        chi_max = maximum(maxlinkdim(ll),maxlinkdim(rr))

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

    end

    return ll, rr, info_iterations

end
