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

    (; opt_method, itermax, truncp, normalization, compute_fidelity, eps_converged) = pm_params
    (; cutoff, maxdim) = truncp

    stopper = PMstopper(pm_params; eps_converged)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    sprevs = ones(length(in_mps)-1, maxlinkdim(in_mps)*max(maxlinkdim(in_mpo_1),maxlinkdim(in_mpo_O)))

    p = Progress(itermax; desc="[PM|$(opt_method)] L=$(length(ll)), cutoff=$(cutoff), maxdim=$(maxdim), normalize=$(normalization)", showspeed=true) 

    info_iterations = Dict(:chis => Int[], :ds2 => Float64[], :logfidelityRRnew => Float64[], :LRdiff => ComplexF64[] )

    for jj = 1:itermax  

        rr_prev = compute_fidelity ? copy(rr) : nothing


        if opt_method == "RTM_LR"
    
            # optimize <LO|1R> -> new |R> 
            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL, truncp)

            # optimize <L1|OR> -> new <L|  
            #TODO: we could be using either the new rr here or the previous rr (in that case should define rr_work = rr before)
            OpsiR = applyn(in_mpo_O, rr)
            OpsiL = applyns(in_mpo_1, ll)  

            _, ll, _ = truncate_sweep(OpsiR, OpsiL, truncp)

        elseif opt_method == "RTM_R"

            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr

        elseif opt_method == "RTM_R_twolayers"

            OpsiR = applyn(in_mpo_1, rr)
            OpsiR = applyn(in_mpo_1, OpsiR)

            OpsiL = applyns(in_mpo_1, rr)  
            OpsiL = applyns(in_mpo_O, OpsiL)  

            rr, _, SVs = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr

        elseif opt_method == "RDM"
        
            ll, _ = tapplys(in_mpo_1, ll; cutoff, maxdim)
            rr, SVs = tapply(in_mpo_1, rr; cutoff, maxdim)

        elseif opt_method == "RDM_R"
   
            rr, SVs = tapply(in_mpo_1, rr; cutoff, maxdim)
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
        else
            normalize_for_overlap!(ll,rr)
        end



        ds2 = max_diff(sprevs, SVs) 
        push!(info_iterations[:ds2], ds2)
        sprevs = SVs

        logfidelityRRnew = compute_fidelity ? logfidelity(rr_prev,rr) : NaN

        push!(info_iterations[:logfidelityRRnew], logfidelityRRnew)
        chimax = max(maxlinkdim(ll),maxlinkdim(rr))
        push!(info_iterations[:chis], chimax)
        
        #maxnormS = maximum([norm(ss) for ss in sjj])

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(chimax)")
        end

        stop, reason = should_stop_ds2!(stopper,ds2)


        if stop
            if reason == :converged
                @info "Converged after $jj steps (ds2=$(ds2) chi=$(chimax))"
            elseif reason == :stuck
                @warn "Iteration stuck after $jj steps (ds2=$(ds2)); stopping."
            end
            break
        end

        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(ds2), logfidelity(<R|Rnew>)=$(logfidelityRRnew)" )])

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

    (; opt_method, itermax, eps_converged, truncp, normalization, compute_fidelity) = pm_params

    # Normalize eps_converged by system size or larger chains will never converge as good...
    eps_converged = eps_converged * length(in_mps)
    stopper = PMstopper(pm_params; eps_converged)
  
    cutoff = truncp.cutoff
    maxdim = truncp.maxdim

    mpslen = length(in_mps)

    ds2s = ComplexF64[]
    ds2 = 0.
    sprevs = fill(1., mpslen-1)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(truncp.cutoff), χmax=$(truncp.maxdim), normalize=$(normalization)", showspeed=true) 

    for jj = 1:itermax

        OpsiL = applyns(in_mpo_L, ll)
        OpsiR = applyn(in_mpo_R, rr)

        llprev = deepcopy(ll)

        if opt_method == "RDM"
            ll = truncate(OpsiL; cutoff, maxdim)
            rr = truncate(OpsiR; cutoff, maxdim)
            sjj = vn_entanglement_entropy(ll)

        else # RTM
            ll, rr, sjj = truncate_sweep(OpsiL, OpsiR, truncp)

        end

        if normalization == "norm"
            ll = normalize(ll)
            rr = normalize(rr)
        elseif normalization == "overlap"
            normalize_for_overlap!(ll,rr)
        end  # otherwise do nothing, norms can blow up 

        #@show overlap_noconj(ll,rr)

        chimax = max(maxlinkdim(ll),maxlinkdim(rr))

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)

        sprevs = sjj

        fidelity_step = compute_fidelity ? logfidelity(ll, llprev) : NaN

        next!(p; showvalues = [(:Info,"[$(jj)] | ds2=$(ds2) | logfidelity(L|Lnew) = $(fidelity_step) | chi=$(maxlinkdim(ll))" )])
        
        stop, reason = should_stop_ds2!(stopper,ds2)

        # --- stopping check ---

        if stop
            if reason == :converged
                @info "Converged after $jj steps (ds2=$(ds2))"
            elseif reason == :stuck
                @warn "Iteration stuck after $jj steps (ds2=$(ds2)); stopping."
            end
            break
        end


    end

    return ll, rr, ds2s

end
