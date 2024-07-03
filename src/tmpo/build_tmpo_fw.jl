""" Builds temporal MPO and starting tMPS guess for *forward evolution only* 
with `nbeta` steps of imaginary time regularization.
Closes with initial state again, so it's a Loschmidt echo type setup.

Returns (tMPO, tMPS) pair.

The structure built (Loschmidt style) after rotation is
```
(left[bottom]_state)---Wβ--Wβ---Wt-Wt-Wt-...-Wt---Wβ--Wβ---(right[top]_state)
                      (nbeta)                     (nbeta)
```
 """
function fw_tMPO(eH::MPO, eHi::MPO, 
    left_state::Vector{<:Number}, 
    right_state::Vector{<:Number},
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

    Nsteps = length(time_sites)

    check_symmetry_itensor_mpo(Wc, ivL, ivR, space_p',space_p)

    rot_links_mpo = [Index(dim(ivL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]
    rot_links_mps = sim(rot_links_mpo)


    left_tensor = ITensor(left_state, space_p)
    right_tensor = ITensor(right_state, space_p')

    # Build tMPO, tMPS with rotation 90degrees: (L,R,P,P') => (P',P,R,L) 
    tMPO = MPO(Nsteps)
    tMPS = MPS(Nsteps)

    for ii = 1:nbeta
        tMPO[ii] = (Wc_im) * delta(ivL_i, time_sites[ii]') * delta(ivR_i, time_sites[ii]) 
        tMPS[ii] = (Wl_im) * delta(ivL_i, time_sites[ii]) 
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPO[ii] = Wc * delta(ivL, time_sites[ii]') * delta(ivR, time_sites[ii]) 
        tMPS[ii] = Wl * delta(ivL, time_sites[ii])

    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPO[ii] = dag(Wc_im) * delta(ivL_i, time_sites[ii]') * delta(ivR_i, time_sites[ii]) 
        tMPS[ii] = dag(Wl_im) * delta(ivL_i, time_sites[ii]) 
    end


    # Contract edges with boundary states, label linkinds
    tMPO[1] *= (left_tensor) * delta(space_p', rot_links_mpo[1]) 
    tMPS[1] *= (left_tensor * delta(space2_p, space_p)) * delta(space2_p', rot_links_mps[1]) 

    for ii = 2:Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links_mpo[ii-1]) * delta(space_p', rot_links_mpo[ii]) 
        tMPS[ii] *= delta(space2_p, rot_links_mps[ii-1]) * delta(space2_p', rot_links_mps[ii]) 
    end

    tMPO[end] *= dag(right_tensor) * delta(space_p, rot_links_mpo[Nsteps-1]) 
    tMPS[end] *= (dag(right_tensor) * delta(space2_p', space_p')) * delta(space2_p, rot_links_mps[Nsteps-1]) 

    return tMPO, tMPS

end

""" Returns (tMPO, tMPS) pair. """
function fw_tMPO(tp::tmpo_params, time_sites::Vector{<:Index})

    eH = build_expH(tp)
    eHi = build_expHim(tp)

    match_siteinds!(eH, eHi)

    fw_tMPO(eH, eHi, tp.bl, tp.tr, tp.nbeta, time_sites)
end

