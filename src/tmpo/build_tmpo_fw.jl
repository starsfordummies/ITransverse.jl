""" Builds temporal MPO and starting tMPS guess for *forward evolution only* 
with `nbeta` steps of imaginary time regularization.
Closes with initial state again, so it's a Loschmidt echo type setup.

Returns (tMPO, tMPS) pair.

The structure built (Loschmidt style is)
```
(init_state)-Wβ-Wβ-Wt-Wt-Wt-...-Wt-Wβ-Wβ-(init_state)
             (nbeta)               (nbeta)
```
 """
function fw_tMPO(eH::MPO, eHi::MPO, 
    init_state::Vector{Number}, 
    fin_state::Vector{Number},
    nbeta::Int, 
    time_sites::Vector{<:Index})

    @assert nbeta < length(time_sites) - 2
    @assert length(eH) == length(eHi) == 3

    if maxlinkdim(eH) != maximum(dims(time_sites))
        @error "Link dimension of unrotated MPO does not match new physical sites: $(maxlinkdim(eH)) vs $(maximum(dims(time_sites)))"
        @assert maxlinkdim(eH) == maximum(dims(time_sites))
    end

    Wl, Wc, _ = eH.data
    Wl_im, Wc_im, _ = eHi.data

    
    space_p = siteind(eH,2)
    space2_p = siteind(eH,1)


    (ivL, ivR) = linkinds(eH)
    (ivL_i, ivR_i) = linkinds(eHi)

    #println(Wc)

    Nsteps = length(time_sites)

    check_symmetry_itensor_mpo(Wc, ivL, ivR, space_p',space_p)

    rot_links = [Index(dim(ivL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]
    rot_links2 = sim(rot_links)


    init_tensor = ITensor(init_state, space_p)
    fin_tensor = ITensor(fin_state, space_p')

    # Build tMPO, tMPS with rotation 90degrees 
    tMPO = MPO(Nsteps)
    tMPS = MPS(Nsteps)

    for ii = 1:nbeta
        tMPO[ii] = dag(Wc_im) * delta(ivL_i, time_sites[ii]) * delta(ivR_i, time_sites[ii]') 
        tMPS[ii] = dag(Wl_im) * delta(ivL_i, time_sites[ii]) 
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPO[ii] = Wc * delta(ivL, time_sites[ii]) * delta(ivR, time_sites[ii]') 
        tMPS[ii] = Wl * delta(ivL, time_sites[ii])

    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPO[ii] = Wc_im * delta(ivL_i, time_sites[ii]) * delta(ivR_i, time_sites[ii]') 
        tMPS[ii] = Wl_im * delta(ivL_i, time_sites[ii]) 
    end


    # Contract edges with init/fin state, label linkinds
    tMPO[1] *= dag(fin_tensor) * delta(space_p, rot_links[1]) 
    tMPS[1] *= dag(fin_tensor) * delta(space_p', space2_p') * delta(space2_p, rot_links2[1]) 

    for ii = 2:Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links[ii-1]) * delta(space_p', rot_links[ii]) 
        tMPS[ii] *= delta(space2_p, rot_links2[ii-1]) * delta(space2_p', rot_links2[ii]) 
    end

    tMPO[Nsteps] *= init_tensor * delta(space_p', rot_links[Nsteps-1]) 
    tMPS[Nsteps] *= init_tensor * delta(space_p, space2_p) * delta(space2_p', rot_links2[Nsteps-1]) 

    return tMPO, tMPS

end

""" Returns (tMPO, tMPS) pair. """
function fw_tMPO(tp::tmpo_params, time_sites::Vector{<:Index})

    eH = build_expH(tp)
    eHi = build_expHim(tp)

    match_siteinds!(eH, eHi)

    fw_tMPO(eH, eHi, tp.init_state, tp.init_state, tp.nbeta, time_sites)
end

