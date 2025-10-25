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

    (; opt_method, itermax, eps_converged, truncp, normalization) = pm_params
    (; cutoff, maxbondim) = truncp

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)


    sprevs = fill(1., length(in_mps)-1)
    LRprev = overlap_noconj(in_mps,in_mps)

    p = Progress(itermax; desc="[PM|$(opt_method)] L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    info_iterations = Dict(:ds2 => ComplexF64[], :logfidelityRRnew => Float64[], :LRdiff => ComplexF64[] )

    for jj = 1:itermax  

        rr_prev = copy(rr)

        # When do we normalize? Here I choose to do it at the beginning of each iteration 

        if normalization == "norm"
            ll = orthogonalize!(ll,1)
            rr = orthogonalize!(rr,1)

            ll = normalize(ll)
            rr = normalize(rr)
        else
            normalize_for_overlap!(ll,rr)
        end


        if opt_method == "RTM_LR"
    
            # optimize <LO|1R> -> new |R> 
            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)

            # optimize <L1|OR> -> new <L|  
            #TODO: we could be using either the new rr here or the previous rr (in that case should define rr_work = rr before)
            OpsiR = applyn(in_mpo_O, rr)
            OpsiL = applyns(in_mpo_1, ll)  

            _, ll, _ = truncate_sweep(OpsiR, OpsiL, truncp)


        elseif opt_method == "RTM_R"

            OpsiR = applyn(in_mpo_1, rr)
            OpsiL = applyns(in_mpo_O, ll)  

            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr

        elseif opt_method == "RTM_R_twolayers"

            OpsiR = applyn(in_mpo_1, rr)
            OpsiR = applyn(in_mpo_1, OpsiR)

            OpsiL = applyns(in_mpo_1, rr)  
            OpsiL = applyns(in_mpo_O, OpsiL)  

            rr, _, sjj = truncate_sweep(OpsiR, OpsiL, truncp)
            ll = rr


        elseif opt_method == "RDM"
        
            ll = applys(in_mpo_1, ll, cutoff=cutoff, maxdim=maxbondim)
            rr = apply(in_mpo_1, rr, cutoff=cutoff, maxdim=maxbondim)

            sjj = vn_entanglement_entropy(rr)

            #@show jj, norm(ll), norm(rr), overlap_noconj(ll,rr)

        elseif opt_method == "RDM_SYMLR"
   
            rr = apply(in_mpo_1, rr, cutoff=cutoff, maxdim=maxbondim)
            sjj = vn_entanglement_entropy(rr)
            ll = rr

        else
            @error "Wrong optimization method: $opt_method"
        end



        LRnew = overlap_noconj(ll,rr)
        push!(info_iterations[:LRdiff], abs(LRnew-LRprev))
        LRprev = LRnew
   
        ds2 = norm(sprevs - sjj)
        push!(info_iterations[:ds2], ds2)
        sprevs = sjj

        logfidelityRRnew = logfidelity(rr_prev,rr)

        push!(info_iterations[:logfidelityRRnew], logfidelityRRnew)
        
        maxnormS = maximum([norm(ss) for ss in sjj])

        if ds2 < eps_converged
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(ds2), logfidelity(<R|Rnew>)=$(logfidelityRRnew) |S|=$(maxnormS)" )])

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

    (; opt_method, itermax, eps_converged, truncp, normalization) = pm_params

    cutoff = truncp.cutoff
    maxdim = truncp.maxbondim

    mpslen = length(in_mps)

    ds2s = ComplexF64[]
    ds2 = 0.
    sprevs = fill(1., mpslen-1)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim))", showspeed=true) 

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
        else
            normalize_for_overlap!(ll,rr)
        end

        #@show overlap_noconj(ll,rr)


        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)

        sprevs = sjj

        fidelity_step = logfidelity(ll, llprev)

        next!(p; showvalues = [(:Info,"[$(jj)] | ds2=$(ds2) | logfidelity(old|new)) = $(fidelity_step)) | chi=$(maxlinkdim(ll))" )])

        if ds2 < eps_converged || fidelity_step > 1e-8
            println("converged after $jj steps")
            break
        end

    end

    return ll, rr, ds2s

end
