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
The most consistent way to do it is probably to enforce that the overlap <L|R> = 1, but in practice normalizing individually
<L|L> = <R|R> = 1 seems to work as well. Another possibility is to normalize the overlap <LO|1R> before truncating for |Rnew>.

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

    (; opt_method, itermax, eps_converged, truncp) = pm_params
    (; cutoff, maxbondim, direction) = truncp

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)


    sprevs = fill(1., length(in_mps)-1)
    LRprev = overlap_noconj(in_mps,in_mps)

    p = Progress(itermax; desc="[PM|$(opt_method)] L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    info_iterations = Dict(:ds2 => ComplexF64[], :RRnew => ComplexF64[], :LRdiff => ComplexF64[] )

    for jj = 1:itermax  

        if opt_method == "RTM_LR"
            
            rr_work = rr
            ll_work = ll 
    
            # optimize <LO|1R> -> new |R> 
            OpsiR = applyn(in_mpo_1, rr_work)
            OpsiL = applyns(in_mpo_O, ll_work)  

            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)

            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)

            # optimize <L1|OR> -> new <L|  
            #TODO: we could be using the new rr here instead of rr_work
            OpsiR = applyn(in_mpo_O, rr_work)
            OpsiL = applyns(in_mpo_1, ll_work)  

            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)

            _, ll, _ = truncate_sweep(OpsiR, OpsiL, truncp)


        elseif opt_method == "RTM_R"
            rr_work = normbyfactor(rr, sqrt(overlap_noconj(rr,rr)))
            #rr_work = normalize(rr)
            #rr_work = rr

            OpsiR = applyn(in_mpo_1, rr_work)
            OpsiL = applyns(in_mpo_O, rr_work)  

            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)
    
            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr

        elseif opt_method == "RTM_R_twolayers"
            rr_work = normbyfactor(rr, sqrt(overlap_noconj(rr,rr)))
            #rr_work = normalize(rr)
            #rr_work = rr

            OpsiR = applyn(in_mpo_1, rr_work)

            OpsiR = applyn(in_mpo_1, OpsiR)

            OpsiL = applyns(in_mpo_1, rr_work)  

            OpsiL = applyns(in_mpo_O, OpsiL)  



            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)
    
            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr

        elseif opt_method == "RTM_R_twolayers_alt"
            rr_work = normbyfactor(rr, sqrt(overlap_noconj(rr,rr)))
            #rr_work = normalize(rr)
            #rr_work = rr

            OpsiR = applyn(in_mpo_1, rr_work)

            OpsiR = applyn(in_mpo_1, OpsiR)

            OpsiL = applyns(in_mpo_O, rr_work)  

            OpsiL = applyns(in_mpo_1, OpsiL)  



            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)
    
            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr


        elseif opt_method == "RDM"
            #rr_work = normbyfactor(rr, sqrt(overlap_noconj(rr,rr)))
            rr_work = normalize(rr)

            rr = apply(in_mpo_1, rr_work, cutoff=cutoff, maxdim=maxbondim)
            ll = rr
            sjj = vn_entanglement_entropy(rr)

        else
            @error "Wrong optimization method: $opt_method"
        end


        # If I cook them separately, likely the overlap will be messed up 
        LRnew = overlap_noconj(ll,rr)
        push!(info_iterations[:LRdiff], abs(LRnew-LRprev))
        LRprev = LRnew
   
        if abs(LRnew) < 1e-6
            @warn "Small overlap $LRnew, watch for trunc error"
        end

        ds2 = norm(sprevs - sjj)
        push!(info_iterations[:ds2], ds2)
        # push!(ds2s, ds2)
        sprevs = sjj

        RRnew = inner(rr_work,rr)/norm(rr)/norm(rr_work)

        push!(info_iterations[:RRnew], RRnew)

        maxnormS = maximum([norm(ss for ss in sjj)])

        if ds2 < eps_converged
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(ds2), <R|Rnew>=1-$(round(1-RRnew,digits=8)) |S|=$(maxnormS)" )])

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

    (; opt_method, itermax, eps_converged, truncp) = pm_params

    mpslen = length(in_mps)

    ds2s = ComplexF64[]
    ds2 = 0.
    sprevs = fill(1., mpslen-1)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 

    for jj = 1:itermax

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        # Take the ll vector from the previous step as new L,R 
        OpsiL = applyns(in_mpo_L, ll)
        OpsiR = applyn(in_mpo_R, rr)

        llprev = deepcopy(ll)
        rrprev = deepcopy(rr)

        # TODO implement different methods according to `opt_method`
        ll, rr, sjj = truncate_normalize_sweep(OpsiL, OpsiR, truncp)

        ds2 = norm(sprevs - sjj)

        #push!(ds2s, ds2)
        push!(ds2s, inner(llprev,ll))
        sprevs = sjj


        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(ll))" )])

        if ds2 < eps_converged
            println("converged after $jj steps")
            break
        end

    end

    return ll, rr, ds2s

end

