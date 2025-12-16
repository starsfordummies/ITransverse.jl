
"""
Given an MPO tensor `W` and the relevant indices,
We rotate our space vectors to the *right* by 90°, ie 

(L,R,P,P') => (P',P,R,L)
```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```
 Convention we stick to for all the following: indices for tMPO WWl before rotations are  L, R, P, P'
"""

"""
Builds *Folded* but **UN-ROTATED** tensors, just W * Wdag and joined indices 
"""

function build_WW(eH::MPO)

    # Same indices for all tensors
    space_phys = Index(dim(siteind(eH,2))^2, "Site,space")
    space_link1 = Index(linkdim(eH,1)^2, "Link,space")
    space_link2 = Index(linkdim(eH,2)^2, "Link,space")

    WWl,       iCwR, iCp, iCps = build_WWl(eH)
    WWl = replaceinds(WWl, (iCwR, iCp, iCps),  (space_link1, space_phys, space_phys'))

    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    WWc = replaceinds(WWc, (iCwL, iCwR, iCp, iCps),  (space_link1, space_link2, space_phys, space_phys'))

    WWr, iCwL,       iCp, iCps = build_WWr(eH)
    WWr = replaceinds(WWr, (iCwL, iCp, iCps),  (space_link2, space_phys, space_phys'))

    return WWl, WWc, WWr,  (space_link1, space_link2, space_phys, space_phys')
end

function build_WW(tp::tMPOParams)
    @info "Building WW tensors using $(tp.expH_func), parameters $(tp.mp)"
    eH = ModelUt(tp).Ut
    build_WW(eH)
end
    
### WWl - WWc - WWr 

""" Builds folded unrotated left edge tensor of the network """ 
function build_WWl(eH_space::MPO)

    Wl = eH_space[1]

    space_p = siteind(eH_space,1)
    vR = linkinds(eH_space)[1]

    # Build W Wdagger - put double prime on Wconj for safety
    WWl = Wl * dag(prime(Wl,2))

    # Combine indices  
    CvR = combiner(vR,vR''; tags="cwR")
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWl = WWl * CvR * Cp * Cps

    #We return indices as well
    iCvR = combinedind(CvR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWl, iCvR, iCp, iCps
end


""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' 
"""
function build_WWc(tp::tMPOParams)
    eH = ModelUt(tp).Ut
    build_WWc(eH)
end

function build_WWc(eH_space::MPO)

    Wc = eH_space[2]

    space_p = siteind(eH_space,2)
    (wL, wR) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj 
    WWc = Wc * dag(prime(Wc,2))

    # Combine indices
    CwL = combiner(wL,wL''; tags="cwL")
    CwR = combiner(wR,wR''; tags="cwR")
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWc = WWc * CwL * CwR * Cp * Cps

    #We return indices as well
    iCwL = combinedind(CwL)
    iCwR = combinedind(CwR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWc, iCwL, iCwR, iCp, iCps
end

""" Builds folded unrotated right edge tensor of the network """ 
function build_WWr(eH_space::MPO)

    LL = length(eH_space)
    Wr = eH_space[end]

    space_p = siteind(eH_space,LL)

    vL = linkinds(eH_space)[LL-1]

    # Build W Wdagger - put double prime on Wconj
    WWr = Wr * dag(prime(Wr,2))

    # Combine indices 
    CvL = combiner(vL,vL''; tags="cwL")
    Cp = combiner(space_p,space_p''; tags="cp")
    Cps = combiner(space_p',space_p'''; tags="cps")

    WWr = WWr * CvL * Cp * Cps

    #We return indices as well
    iCvL = combinedind(CvL)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWr, iCvL, iCp, iCps
end
