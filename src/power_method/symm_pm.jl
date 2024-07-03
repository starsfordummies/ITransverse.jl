"""
Power method for *symmetric* case: takes as input a single MPS |L>,
    applies the `in_mpo` O and optimizes the overlap (LO|OL) , where LO is *not* conjugated.
    Works eg. for unfolded Loschmidt echo type setups, where `in_mpo` is the transfer matrix 
    and truncating over (LO|OL) does not give trivial results.
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams)

    itermax = pm_params.itermax
    converged_ds2 = pm_params.ds2_converged

    (; cutoff, maxbondim, ortho_method) = pm_params.trunc_params
    # cutoff = pm_params.trunc_params.cutoff
    # maxbondim = pm_params.trunc_params.maxbondim
    # method = pm_params.trunc_params.ortho_method

    # normalize the vector to get a good starting point

    psi_ortho = normalize(in_mps)

    ds2s = Float64[]
    ds2 = 0. 
    sprevs = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="L=$(length(in_mps)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 


    maxbondim = 20 

    for jj = 1:itermax

        if pm_params.increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim,pm_params.trunc_params.maxbondim])
        else
            maxbondim = pm_params.trunc_params.maxbondim
        end

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        psi = applyn(in_mpo, psi_ortho)
        psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method=ortho_method)
        
        # Here it's actually important to normalize after each iteration 
        psi_ortho[1] /= sqrt(overlap)
        #@show overlap

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(psi))" )])

        if ds2 < converged_ds2
            break
        end


    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(psi_ortho))")

    # nicer link labels
    replace_linkinds!(psi_ortho, "Link,rotl=")
    return psi_ortho, ds2s

end



"""
Power method using SVD compression: takes as input a single MPS |L>,
    applies O and optimizes the overlap <LObar|OL>  (ie. LO conjugate) - in practice 
    this is the standard truncation using SVDs/RDMs, or the standard temporal entropy 

Truncation params are in pm_params
"""
function powermethod_sym_rdm(in_mps::MPS, in_mpo::MPO, pm_params::Dict)

    
    itermax = pm_params[:itermax]
    SVD_cutoff = pm_params[:SVD_cutoff] 
    maxbondim = pm_params[:maxbondim]

    # normalize the vector to get a good starting point?

    ll = normalize(in_mps)

    ds2s = [0.]
    ds2 = fill(0., length(in_mps)-1)
    sprevs = fill(1., length(in_mps)-1)

    for jj in range(1,itermax)

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


