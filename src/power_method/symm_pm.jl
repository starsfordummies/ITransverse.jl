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
        
        if ortho_method == "RDM"
            psi_ortho = apply(in_mpo, psi,  alg="naive" , truncate=true, cutoff=SVD_cutoff, maxim=maxbondim)
        else
            psi = applyn(in_mpo, psi_ortho)
            psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method=ortho_method)
        end
            

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
