
"""
Power method 

    Starts with |in_mps>, builds <L| = <in_mps|in_mpo_L and |R> = in_mpo_R|in_mps>
    and optimizes them iteratively. 
    Note that at each step, there is a single optimization which returns a new set Lnew, Rnew
    which are used as starting point for the next step. 

    So this is only if we don't put in extra operators and the like. For that case, use `powermethod_fold`

Truncation params are in pm_params
"""
function powermethod(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::ppm_params; flip_R::Bool=false)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    method = pm_params.ortho_method

    mpslen = length(in_mps)

    ds2s = ComplexF64[]
    ds2 = 0.
    sprevs = fill(1., mpslen-1)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    # TODO CHECK 
    if flip_R
        in_mpo_R = swapprime(in_mpo_R, 0, 1, "Site")
    end

    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 

    for jj = 1:itermax

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        # Take the ll vector from the previous step as new L,R 
        OpsiL = apply(in_mpo_L, ll,  alg="naive", truncate=false)
        OpsiR = apply(in_mpo_R, rr,  alg="naive", truncate=false)

        llprev = deepcopy(ll)
        rrprev = deepcopy(rr)

        ll, rr, sjj = truncate_normalize_sweep(OpsiL, OpsiR, cutoff=cutoff, method=method, chi_max=maxbondim)

        # TODO: this is costly - maybe not necessary if all we care for is convergence
        #sjj = generalized_entropy(ll,rr)

        # DEBUG: plot the evolution of the entropies
        if pm_params.plot_s
            display(plot(real(sjj),label=jj,legend=:outertopright))
        end

        ds2 = norm(sprevs - sjj)

        
        #push!(ds2s, ds2)
        push!(ds2s, inner(llprev,ll))
        sprevs = sjj

        #println("$(jj): [$(maxlinkdim(OpsiL)),$(maxlinkdim(OpsiR))] => $(maxlinkdim(ll)) , ds2 = $(ds2), <L|R> = $(overlap_noconj(ll,rr))")

        @show inner(OpsiL, ll)
        @show inner(OpsiR, rr)
        @show inner(llprev,ll)
        @show inner(rrprev,rr)

        # this maybe helps keeping memory usage low on cluster(?)
        # https://itensor.discourse.group/t/large-amount-of-memory-used-when-dmrg-runs-on-cluster/1045/4
        #GC.gc()

        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(ll))" )])


        if ds2 < 1e-10
            println("converged after $jj steps")
            break
        end

    end

    return ll, rr, ds2s

end


""" Power method with convergence criterion based on overlap <Lprevious step|L> """
function powermethod_converge_norm(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::ppm_params; flip_R::Bool=false)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    method = pm_params.ortho_method

    mpslen = length(in_mps)


    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    # TODO CHECK 
    if flip_R
        in_mpo_R = swapprime(in_mpo_R, 0, 1, "Site")
    end

    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 

    overlaps = ComplexF64[]
    ov2prev = 666.0

    for jj = 1:itermax

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        # Take the ll vector from the previous step as new L,R 
        OpsiL = apply(in_mpo_L, ll,  alg="naive", truncate=false)
        OpsiR = apply(in_mpo_R, rr,  alg="naive", truncate=false)

        llprev = deepcopy(ll)
        rrprev = deepcopy(rr)

        ll, rr, sjj = truncate_normalize_sweep(OpsiL, OpsiR, cutoff=cutoff, method=method, chi_max=maxbondim)

        ov2 = sqrt(abs2(inner(llprev,ll)) + abs2(inner(rrprev,rr))) 
        push!(overlaps, ov2 )
        

        next!(p; showvalues = [(:Info,"[$(jj)] Δnorm=$(ov2prev-ov2), chi=$(maxlinkdim(ll))" )])


        if abs(ov2prev-ov2) < 1e-5
            println("converged after $jj steps")
            break
        end
        ov2prev = ov2


    end

    return ll, rr, overlaps

end


""" Power method with convergence criterion based on overlap <Lprevious step|L> """
function powermethod_converge_eig(in_mps::MPS, in_mpo_L::MPO, in_mpo_R::MPO, pm_params::ppm_params; flip_R::Bool=false)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    method = pm_params.ortho_method

    mpslen = length(in_mps)


    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    # TODO CHECK 
    if flip_R
        in_mpo_R = swapprime(in_mpo_R, 0, 1, "Site")
    end

    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 

    lambdas = []
    λl_prev = 0.0
    λr_prev = 0.0

    for jj = 1:itermax

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        # Take the ll vector from the previous step as new L,R 
        OpsiL = apply(in_mpo_L, ll,  alg="naive", truncate=false)
        OpsiR = apply(in_mpo_R, rr,  alg="naive", truncate=false)

        llprev = deepcopy(ll)
        rrprev = deepcopy(rr)

        ll, rr, sjj = truncate_normalize_sweep(OpsiL, OpsiR, cutoff=cutoff, method=method, chi_max=maxbondim)

        λl = norm(OpsiL)/norm(llprev)
        λr = norm(OpsiR)/norm(rrprev)

        λl_alt = norm_gen(OpsiL)/norm_gen(llprev)
        λr_alt = norm_gen(OpsiR)/norm_gen(rrprev)

        push!(lambdas, [λl, λr, λl_alt, λr_alt] )
        
        next!(p; showvalues = [(:Info,"[$(jj)] ΔλL=$(λl_prev-λl), ΔλR=$(λr_prev-λr) chi=$(maxlinkdim(ll))" )])


        if abs(λl_prev-λl) +  abs(λr_prev-λr)  < 1e-5
            println("converged after $jj steps, $(abs(λl_prev-λl)), $(abs(λr_prev-λr))")
            break
        end

        λl_prev = λl
        λr_prev = λr

    end

    return ll, rr, lambdas

end





"""
Power method developed for the folding algorithm, takes as input TWO mpos, 
    one meant to be with an additional operator (X) 

Neither of the input MPOs needs to have the L-R indices swapped, we do it in here already.
    
The algorithm makes *two* updates, first it computes <L1|OR> and updates <Lnew|, 
    then computes <LO|1R> and computes |Rnew>

Truncation params are in pm_params
"""
function powermethod_fold(in_mps::MPS, in_mpo_1::MPO, in_mpo_X::MPO, pm_params::ppm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    converged_ds2 = pm_params.ds2_converged


    # normalize the vector to get a good starting point?
    #ll = normalize(in_mps)
    #rr = normalize(in_mps)

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    @show overlap_noconj(ll,rr)
    
    # should we also normalize the MPO !!?
    #normalize!(in_mpo)

    ds2s = Float64[]
    ds2 = 0. # fill(0., length(in_mps)-1)  #??
    sprevs = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    # @show check_symmetry_itensor(in_mpo_1[1], siteinds(in_mpo_1,1) )
    # @show check_symmetry_itensor(in_mpo_X[1], siteinds(in_mpo_X,1) )

    # @show check_symmetry_itensor_mpo(in_mpo_1[4])
    # @show check_symmetry_itensor_mpo(in_mpo_X[4])

    # @show check_symmetry_itensor(in_mpo_1[end], siteinds(in_mpo_1, length(ll)))
    # @show check_symmetry_itensor(in_mpo_X[end], siteinds(in_mpo_X, length(ll)))

    for jj = 1:itermax  

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        ll_work = deepcopy(ll)

        OpsiL = apply(in_mpo_1, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_X, 0, 1, "Site"), rr,  alg="naive", truncate=false)  

        ll, _r, sjj = truncate_normalize_sweep(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        #check_gencan_left_sym_phipsi(ll, _r, true)


        #@show overlap_noconj(OpsiL, OpsiR)

        OpsiL = apply(in_mpo_X, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr,  alg="naive", truncate=false)  


        # DEBUG
        #check_equivalence_svd(OpsiL, OpsiR)
        
        _l, rr, sjj = truncate_normalize_sweep(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        #check_gencan_left_sym_phipsi(_l,rr, true)

        # ! we should be sure about symmetry ..
        #rr = deepcopy(ll)

        # @show matrix(ll[1])
        # @show matrix(rr[1])

        # @show matrix(ll[1])-matrix(rr[1])
        #@show overlap_noconj(OpsiL, OpsiR)

        # If I cook them separately, likely the overlap will be messed up 
        overl = overlap_noconj(ll,rr)
        #@show overl 

        #@show(norm(ll[1]))
        #@show(norm(rr[1]))

        #@show norm(matrix(ll[1])-matrix(rr[1]))

        #@show norm(matrix(ll[end])-matrix(rr[end]))

        if abs(overl) < 0.01
            @warn "Small overlap $overl, watch for trunc error"
        end

        # normalize overlap to 1 at each step 
        ll[end] =  ll[end] /sqrt(overl)
        rr[end] =  rr[end] /sqrt(overl)

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        #println("$(jj): [$(maxlinkdim(OpsiL)),$(maxlinkdim(OpsiR))] => $(maxlinkdim(ll)) , ds2 = $(ds2), <L|R> = $(overlap_noconj(ll,rr))")

        # this maybe helps keeping memory usage low on cluster(?)
        # https://itensor.discourse.group/t/large-amount-of-memory-used-when-dmrg-runs-on-cluster/1045/4
        #GC.gc()

        if ds2 < converged_ds2
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        #next!(p; showvalues = [(:ds2,ds2), (:chi,(maxlinkdim(ll)))])
        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(ll))" )])

    end


    return ll, rr, ds2s

end