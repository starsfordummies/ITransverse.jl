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
function powermethod_sym(in_mps::MPS, in_mpo::MPO, pm_params::PMParams; fast::Bool=false)

    (; itermax, eps_converged, opt_method, truncp, increase_chi, normalization, compute_fidelity) = pm_params
    (; cutoff, maxbondim) = truncp

    # Normalize eps_converged by system size or larger chains will never converge as good...
    eps_converged = eps_converged * length(in_mps)
  
    # normalize initial boundary for stability
    psi_ortho = normalize(in_mps)

    ds2s = [] 
    ds2 = 0. 
    sprevs = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="[Symmetric PM|$(opt_method)] L=$(length(in_mps)), cutoff=$(cutoff), χmax=$(maxbondim), normalize=$(normalization))", showspeed=true) 

    max_chi = maxbondim
    maxbondim = 20 

    for jj = 1:itermax

        if compute_fidelity
            psi_prev = copy(psi_ortho)
            prev_norm = norm(psi_prev)
        end


        if increase_chi
            maxbondim += 2
            maxbondim = minimum([maxbondim, max_chi])
        else
            maxbondim = max_chi
        end

        if opt_method == "RDM"
            psi_ortho = apply(in_mpo, psi_ortho; cutoff=cutoff, maxdim=maxbondim)
            overlap = overlap_noconj(psi_ortho, psi_ortho)
            sjj = vn_entanglement_entropy(psi_ortho)
        elseif opt_method == "RTM"
            # if jj % 200 == 0 
            #     psi_ortho = apply(in_mpo, psi_ortho; maxdim=maxbondim)
            #     psi_ortho = apply(in_mpo, psi_ortho; maxdim=maxbondim)
            #     psi_ortho = apply(in_mpo, psi_ortho; maxdim=maxbondim)
            #     psi_ortho = apply(in_mpo, psi_ortho; maxdim=maxbondim)
            # end
            psi = applyn(in_mpo, psi_ortho)
            psi_ortho, sjj, overlap = truncate_rsweep_sym(psi; cutoff=cutoff, chi_max=maxbondim, method="SVD", fast)
        elseif opt_method == "RTMRDM"
            if jj == div(itermax,2)
                #increase_chi = false
                max_chi = maxlinkdim(psi_ortho)+4 # give it some room for adjustment
                opt_method = "RDM"
                @info "$(jj) - Changing method to RDM "
                overlap = overlap_noconj(psi_ortho, psi_ortho)
                sjj = vn_entanglement_entropy(psi_ortho)
            else # do RTM
                psi = applyn(in_mpo, psi_ortho)
                psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method="SVD")
            end
        #  TODO this can be less accurate
        elseif opt_method == "RTM_EIG"
            psi = applyn(in_mpo, psi_ortho)
            psi_ortho, sjj, overlap = truncate_rsweep_sym(psi, cutoff=cutoff, chi_max=maxbondim, method="EIG")
        else
            @error "Specify a valid opt_method: RDM|RTM|..."
        end
            

        if normalization == "norm"
            #orthogonalize!(psi_ortho,1)
            normalize!(psi_ortho)
        elseif normalization == "overlap"
            # normalize so that <L|R> = 1 
            psi_ortho = psi_ortho / sqrt(overlap)
        end # otherwise we do nothing - norm can blow up! 

        fidelity = if compute_fidelity 
           abs( log(abs(inner(psi_ortho, psi_prev))) - log(norm(psi_ortho)) - log(norm(psi_prev)) ) / length(psi_ortho)
        else 
            NaN
        end

        ds2 = norm(sprevs - sjj)
        push!(ds2s, [ds2, fidelity])

        sprevs = sjj

        next!(p; showvalues = [(:Info,"[$(jj)] ds2=$(ds2), <R|Rprev> = $(fidelity), chi=$(maxlinkdim(psi_ortho))" )])

        # Check convergence 
        if ds2 < eps_converged
            break
        end

    end

    println("Stopped after $(length(ds2s)) steps, final ds^2 = $(ds2s[end]), chimax=$(maxlinkdim(psi_ortho))")

    # nicer link labels
    replace_linkinds!(psi_ortho, "Link,rotl=")
    return psi_ortho, ds2s

end
