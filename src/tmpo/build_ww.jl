
"""
We rotate our space vectors to the *right* by 90°, ie 
(L,R,P,P') => (P',P,R,L)
```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```
"""

function rotate_90clockwise(W::ITensor; L=nothing, R=nothing, P=nothing, Ps=nothing)
    
    #xLtP = Index(dim(L),"time,site")
    xPtR = Index(dim(P),"time,virtR")
    xPstL = Index(dim(P),"time,virtL")

    if !isnothing(L)
        xLtP = Index(dim(L),"time,site")
        W *= delta(L, xLtP')
        if !isnothing(R)
            W *= delta(R, xLtP)
        end
    elseif !isnothing(R)
        xLtP = Index(dim(R),"time,site")
        W *= delta(R, xLtP)
    end

    W *= delta(Ps, xPtR)
    W *= delta(P, xPstL)

    return W
end


""" Convention we stick to for all the following: indices for tMPO WWl before rotations are  L, R, P, P' """
function build_WW(tp::tmpo_params)
    eH = build_expH(tp)
    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    WWl, iCwR, iCp, iCps = build_WWl(eH)
    WWr, iCwL, iCp, iCps = build_WWr(eH)

    return WWl, WWc, WWr
end

    

""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' """

function build_WWc(tp::tmpo_params)
    eH = build_expH(tp)
    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
end

function build_WWc(eH_space::MPO)

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
function build_WWr(eH_space::MPO)

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
function build_WWl(eH_space::MPO)

    Wl, _, _ = eH_space.data

    space_p = siteind(eH_space,1)

    (vR, _) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWl = Wl * dag(prime(Wl,2))

    # Combine indices appropriately 
    CvR = combiner(vR,vR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cps = combiner(space_p',space_p''; tags="cps")
    Cp = combiner(space_p,space_p'''; tags="cp")

    WWl = WWl * CvR * Cp * Cps

    #Return indices as well
    iCvR = combinedind(CvR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWl, iCvR, iCp, iCps
end


""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution rotated 90deg clockwise"""
struct FoldtMPOBlocks
    WWl::ITensor
    WWc::ITensor
    WWr::ITensor
    rho0::ITensor
    tp::tmpo_params

    function FoldtMPOBlocks(tp::tmpo_params, init_state::Vector{<:Number} = tp.bl) 

        WWl, WWc, WWr = build_WW(tp::tmpo_params)

        R, P, Ps = inds(WWl)
        WWl = rotate_90clockwise(WWl; R,P,Ps)
        L, R, P, Ps = inds(WWc)
        WWc = rotate_90clockwise(WWc; L,R,P,Ps)
        L, P, Ps = inds(WWr)
        WWr = rotate_90clockwise(WWr; L,P,Ps)

        # Match the inds of all three tensors FIXME do we want this? 
        il = (Index(1), inds(WWl)...) 
        WWl = replaceinds(WWl, il, inds(WWc))
        ir = (ind(WWr,1), Index(1), ind(WWr,2),ind(WWr,3))
        WWr = replaceinds(WWr, ir, inds(WWc))

        # We can accept either an initial state or initial folded state (DM)
        if length(init_state) == dim(P)
            rho0 = ITensor(init_state, Index(dim(P),"virt,time,rho0"))
        elseif length(init_state) == dim(P) ÷ 2
            rho0 = (init_state) * (init_state')
            rho0 = ITensor(rho0, Index(dim(P),"virt,time,rho0"))
        else
            @error "Dimension of init_state is $(length(init_state)) vs linkdim $(dim(P))"
            rho0 = ITensor(0)
        end

        return new(WWl, WWc, WWr, rho0, tp)
    end


end
