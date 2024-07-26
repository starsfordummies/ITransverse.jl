"""
Power method for *symmetric* case: takes as input a single MPS |L>,
    applies the `in_mpo` O and optimizes the overlap (LO|OL) , where LO is *not* conjugated.
    Works eg. for unfolded Loschmidt echo type setups, where `in_mpo` is the transfer matrix 
    and truncating over (LO|OL) does not give trivial results.
    We can nevertheless specify in `pm_params` the `opt_method = RDM` if we want to use the usual truncation
    based on the reduced density matrix (which boils down to SVD of the Left vector)
    For the RTM truncation, since the problem is symmetric, we can choose to truncate over the SV of 
    the symmetric environments (`opt_method=RTM`), which are computed in a Autonne-Takagi form, 
    or compute the symmetric eigenvalue problem of the RTM   (`opt_method=RTM_EIG`)
    and truncate over the (complex, so be mindful..) eigenvalues of the RTM. 
"""
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams)

    (; itermax, eps_converged, opt_method, truncp) = pm_params
    (; cutoff, maxbondim) = truncp
  
    # normalize the vector to get a good starting point

    psi_ortho = normalize(in_mps)

    ds2s = Float64[]
    ds2 = 0. 
    sprevs = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="[Symmetric PM|$(opt_method)] L=$(length(in_mps)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 


    maxbondim = 20 

    for jj = 1:itermax

        if pm_params.increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim,pm_params.truncp.maxbondim])
        else
            maxbondim = pm_params.truncp.maxbondim
        end

        # Note that ITensors does the apply on the MPS/MPO legs with the SAME label, eg. p-p 
        # and then unprimes the p' leg. 
        
        if opt_method == "RDM"
            psi_ortho = apply(in_mpo, psi_ortho,  alg="naive" , truncate=true, cutoff=cutoff, maxdim=maxbondim)
            overlap = overlap_noconj(psi_ortho, psi_ortho)
            sjj = vn_entanglement_entropy(psi_ortho)
        elseif opt_method == "RTM"
            psi = applyn(in_mpo, psi_ortho)
            psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method="SVD")
        elseif opt_method == "RTM_EIG"
            psi = applyn(in_mpo, psi_ortho)
            psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method="EIG")
        else
            @error "Specify a valid opt_method: RDM|RTM|RTM_EIG"
        end
            

        # Here it's actually important to normalize after each iteration 
        psi_ortho[1] /= sqrt(overlap)
        #@show overlap

        ds2 = norm(sprevs - sjj)
        push!(ds2s, ds2)
        sprevs = sjj

        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), chi=$(maxlinkdim(psi_ortho))" )])

        if ds2 < eps_converged
            break
        end


    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(psi_ortho))")

    # nicer link labels
    replace_linkinds!(psi_ortho, "Link,rotl=")
    return psi_ortho, ds2s

end
