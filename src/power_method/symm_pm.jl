
"""
Power method for SYMMETRIC case: takes as input a single MPS |L>,
    applies the in_mpo O and optimizes the overlap (LO|OL) , where LO is *not* conjugated

Truncation params are in pm_params
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::ppm_params)

    
    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    converged_ds2 = pm_params.ds2_converged
    method = pm_params.ortho_method


    # normalize the vector to get a good starting point?

    ll = normalize(in_mps)

    ds2s = Float64[]
    ds2 = 0. # fill(0., length(in_mps)-1)  #??
    sprevs = fill(1., length(in_mps)-1)

    #@showprogress desc="ds2=$ds2 chi=$(maxlinkdim(ll))" 
    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 


    maxbondim = 10

    for jj = 1:itermax

        # maxbondim += 2
        # maxbondim = minimum([maxbondim,pm_params.maxbondim])

        if pm_params.increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim,pm_params.maxbondim])
        else
            maxbondim = pm_params.maxbondim
        end

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        #OpsiL = apply(in_mpo, ll,  alg="naive", truncate=false)
        #ll, sjj = truncate_normalize_sweep_sym(OpsiL, svd_cutoff=SVD_cutoff, chi_max=maxbondim)

        ll = apply(in_mpo, ll,  alg="naive", truncate=false)
        sjj = truncate_normalize_sweep_sym!(ll, svd_cutoff=cutoff, chi_max=maxbondim, method=method)
        
        #@show linkinds(ll)


        #newInds = ["v" => "l=$i" for i in 1:length(ll)-1]
        #replacetags!(ll, "v" => "Link", newInds... )
        # TODO not implemented yet 
        #sjj = truncate_normalize_sweep_sym_ite!(ll, svd_cutoff=cutoff, chi_max=maxbondim)

        #@show length(sprevs)
        #@show sjj
        #@show length(sjj)

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        if pm_params.verbose
            println("$(jj): $(maxlinkdim(OpsiL)) => $(maxlinkdim(ll)) , ds2 = $(ds2), <L|R> = $(overlap_noconj(ll,ll))")
        end

        if pm_params.plot_s
            templot =  plot(real(sjj),label=jj,legend=:outertopright)
        end

        #next!(p; showvalues = [(:jj,jj), (:ds2,ds2), (:chi,(maxlinkdim(ll)))])
        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(ll))" )])

        #next!(p; showvalues = ["ds2= $ds2, chimax=$(maxlinkdim(ll))"])

        # TODO: this is costly - maybe not necessary if all we care for is convergence
        #sjj = generalized_entropy_symmetric(ll)

        # DEBUG: plot the evolution of the entropies
        #display(plot(real(sjj),label=jj,legend=:outertopright))
        if pm_params.plot_s
            show(plot(real(sjj),label=jj,legend=:outertopright))
        end


        # this maybe helps keeping memory usage low on cluster(?)
        # https://itensor.discourse.group/t/large-amount-of-memory-used-when-dmrg-runs-on-cluster/1045/4
        #GC.gc()


        if ds2 < converged_ds2
            break
        end


    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(ll))")

    return ll, ds2s

end


"""
Power method for SYMMETRIC case: takes as input a single MPS |L>,
    applies the in_mpo O and optimizes the overlap (LO|OL) , where LO is *not* conjugated

    Also computes norms evolution 
Truncation params are in pm_params
"""
function powermethod_sym_norms(in_mps::MPS, in_mpo::MPO, pm_params::ppm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim
    converged_ds2 = pm_params.ds2_converged
    method = pm_params.ortho_method


    # normalize the vector to get a good starting point?

    ll = normalize(in_mps)

    ds2s = Float64[]
    ds2 = 0. # fill(0., length(in_mps)-1)  #??
    sprevs = fill(1., length(in_mps)-1)

    #@showprogress desc="ds2=$ds2 chi=$(maxlinkdim(ll))" 
    p = Progress(itermax; showspeed=true)  #barlen=40
    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(pm_params.maxbondim))", showspeed=true) 


    maxbondim = 10

    for jj = 1:itermax

        if pm_params.increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim,pm_params.maxbondim])
        else
            maxbondim = pm_params.maxbondim
        end

        ll_new = apply(in_mpo, ll,  alg="naive", truncate=false)

        # overlap before trunc should give the dominant eigenvector (assuming we started from gen. normalized states)
        dominant_eig = overlap_noconj(ll,ll_new)

        sjj = truncate_normalize_sweep_sym!(ll_new, svd_cutoff=cutoff, chi_max=maxbondim, method=method)
        trunc_error = overlap_noconj(ll,ll_new)

        ll = ll_new 

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        if pm_params.plot_s
            templot =  plot(real(sjj),label=jj,legend=:outertopright)
        end

        #next!(p; showvalues = [(:jj,jj), (:ds2,ds2), (:chi,(maxlinkdim(ll)))])
        next!(p; showvalues = [(:Info,"[$(jj)] χ=$(maxlinkdim(ll)) | ΔS=$(round(ds2;digits=5)) | τ0 = $(round(dominant_eig,digits=5)) | trunc = $(round(trunc_error,digits=6))" )])

        #next!(p; showvalues = ["ds2= $ds2, chimax=$(maxlinkdim(ll))"])

        # TODO: this is costly - maybe not necessary if all we care for is convergence
        #sjj = generalized_entropy_symmetric(ll)

        # DEBUG: plot the evolution of the entropies
        #display(plot(real(sjj),label=jj,legend=:outertopright))
        if pm_params.plot_s
            show(plot(real(sjj),label=jj,legend=:outertopright))
        end


        if ds2 < converged_ds2
            break
        end


    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(ll))")

    return ll, ds2s

end



"""
Power method using SVD compression: takes as input a single MPS |L>,
    applies O and optimizes the overlap <LObar|OL>  (ie. LO conjugate) - in practice 
    this is the standard truncation using SVDs/RDMs

Truncation params are in pm_params
"""
function powermethod_sym_rdm(in_mps::MPS, in_mpo::MPO, pm_params::Dict)

    
    itermax = pm_params[:itermax]
    SVD_cutoff = pm_params[:SVD_cutoff] 
    maxbondim = pm_params[:maxbondim]

    # normalize the vector to get a good starting point?

    ll = normalize(in_mps)

    # should we also normalize the MPO !!?
    #normalize!(in_mpo)


    ds2s = [0.]
    ds2 = fill(0., length(in_mps)-1)
    sprevs = fill(1., length(in_mps)-1)

    for jj in range(1,itermax)

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        ll = apply(in_mpo, ll,  alg="naive" , truncate=true, cutoff=SVD_cutoff, maxim=maxbondim)

        sjj = vn_entanglement_entropy(ll)

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        println("$(jj): => $(maxlinkdim(ll)) , ds2 = $(ds2), <L|R> = $(overlap_noconj(ll,ll))")


        if ds2 < 1e-10
            println("converged after $jj steps")
        break

    end

    end


    return ll, ds2s

end


