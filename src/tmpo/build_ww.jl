

""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' """
function build_WWc(eH_space)

    _, Wc, _ = eH_space.data

    space_p = siteind(eH_space,2)

    (wL, wR) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWc = Wc * dag(prime(Wc,2))

    # Combine indices appropriately 
    CwL = combiner(wL,wL''; tags="cwL")
    CwR = combiner(wR,wR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cp = combiner(space_p,space_p'''; tags="cp")
    Cps = combiner(space_p',space_p''; tags="cps")

    WWc = WWc * CwL * CwR * Cp * Cps

    #Return indices as well
    iCwL = combinedind(CwL)
    iCwR = combinedind(CwR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWc, iCwL, iCwR, iCp, iCps
end

""" Builds folded unrotated right edge tensor of the network """ 
function build_WWr(eH_space)

    _, _, Wr = eH_space.data

    space_p = siteind(eH_space,3)

    (_, vL) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWr = Wr * dag(prime(Wr,2))

    # Combine indices appropriately 
    CvL = combiner(vL,vL''; tags="cwL")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cps = combiner(space_p',space_p''; tags="cps")
    Cp = combiner(space_p,space_p'''; tags="cp")

    WWr = WWr * CvL * Cp * Cps

    #Return indices as well
    iCvL = combinedind(CvL)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWr, iCvL, iCp, iCps
end


""" Builds folded unrotated right edge tensor of the network """ 
function build_WWl(eH_space)

    Wl, _, _ = eH_space.data

    space_p = siteind(eH_space,3)

    (vR, _) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWl = Wl * dag(prime(Wl,2))

    # Combine indices appropriately 
    CvR = combiner(vR,vR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cps = combiner(space_p',space_p''; tags="cps")
    Cp = combiner(space_p,space_p'''; tags="cp")

    WWr = WWr * CvR * Cp * Cps

    #Return indices as well
    iCvR = combinedind(CvR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWl, iCvR, iCp, iCps
end




"""
We rotate our space vectors to the *right* by 90°, ie 

```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```
"""

function rotate_90clock(W::ITensor, L=nothing, R=nothing, P=nothing, Ps=nothing)
    @assert dim(L) == dim(R)
    xLtP = Index(dim(L),"time,site")
    xPtR = Index(dim(P),"time,virtR")
    xPstL = Index(dim(P),"time,virtL")


    W *= delta(L, xLtP)
    W *= delta(R, xLtP')
    W *= delta(P, xPtR)
    W *= delta(Ps, xPstL)

    return W
end