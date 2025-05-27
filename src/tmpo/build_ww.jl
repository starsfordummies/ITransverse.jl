
"""
Given an MPO tensor `W` and the relevant indices,
We rotate our space vectors to the *right* by 90°, ie 

(L,R,P,P') => (P',P,R,L)
```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```
"""

""" Convention we stick to for all the following: indices for tMPO WWl before rotations are  L, R, P, P'
Builds *Folded* but ***UN-ROTATED*** tensors, just W * Wdag and joined indices """
function build_WW(eH::MPO)

    # Same indices for all tensors
    space_phys = Index(dim(siteind(eH,2))^2, "Site,space")
    space_vleft = Index(linkdim(eH,1)^2, "Link,space")
    space_vright = Index(linkdim(eH,2)^2, "Link,space")

    WWl,       iCwR, iCp, iCps = build_WWl(eH)
    WWl = replaceinds(WWl, (iCwR, iCp, iCps),  (space_vright, space_phys, space_phys'))

    WWr, iCwL,       iCp, iCps = build_WWr(eH)
    WWr = replaceinds(WWr, (iCwL, iCp, iCps),  (space_vleft, space_phys, space_phys'))

    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    #check_symmetry_itensor_mpo(WWc, iCwL, iCwR, iCp, iCps)
    WWc = replaceinds(WWc, (iCwL, iCwR, iCp, iCps),  (space_vleft, space_vright, space_phys, space_phys'))

    return WWl, WWc, WWr,  (space_vleft, space_vright, space_phys, space_phys')
end

function build_WW(tp::tMPOParams)
    @info "Building WW tensors using $(tp.expH_func), parameters $(tp.mp)"
    eH = build_expH(tp)
    build_WW(eH)
end
    

""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' 
"""
function build_WWc(tp::tMPOParams)
    eH = build_expH(tp)
    build_WWc(eH)
end

function build_WWc(eH_space::MPO)

    #_, Wc, _ = eH_space.data
    Wc = eH_space[2]

    space_p = siteind(eH_space,2)

    (wL, wR) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWc = Wc * dag(prime(Wc,2))

    # Combine indices appropriately 
    CwL = combiner(wL,wL''; tags="cwL")
    CwR = combiner(wR,wR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWc = WWc * CwL * CwR * Cp * Cps

    #Return indices as well
    iCwL = combinedind(CwL)
    iCwR = combinedind(CwR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWc, iCwL, iCwR, iCp, iCps
end

""" Builds folded unrotated right edge tensor of the network """ 
function build_WWr(eH_space::MPO)

    LL = length(eH_space)
    #_, _, Wr = eH_space.data
    Wr = eH_space[end]

    space_p = siteind(eH_space,LL)

    vL = linkinds(eH_space)[LL-1]

    # Build W Wdagger - put double prime on Wconj for safety
    WWr = Wr * dag(prime(Wr,2))

    # Combine indices appropriately 
    CvL = combiner(vL,vL''; tags="cwL")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWr = WWr * CvL * Cp * Cps

    #Return indices as well
    iCvL = combinedind(CvL)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWr, iCvL, iCp, iCps
end


""" Builds folded unrotated left edge tensor of the network """ 
function build_WWl(eH_space::MPO)

    #Wl, _, _ = eH_space.data
    Wl = eH_space[1]

    space_p = siteind(eH_space,1)

    vR = linkinds(eH_space)[1]

    # Build W Wdagger - put double prime on Wconj for safety
    WWl = Wl * dag(prime(Wl,2))

    # Combine indices appropriately 
    CvR = combiner(vR,vR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWl = WWl * CvR * Cp * Cps

    #Return indices as well
    iCvR = combinedind(CvR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWl, iCvR, iCp, iCps
end

