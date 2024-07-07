
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
function rotate_90clockwise(W::ITensor; L=nothing, R=nothing, P=nothing, Ps=nothing)
    
    xPtR = Index(dim(P),"time,virtR")
    xPstL = Index(dim(P),"time,virtL")
    xLtP = Index(1)

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

    rotated_inds = Dict(:L => xPstL, :R => xPtR, :P => xLtP, :Ps => xLtP' )

    # TODO Permute so we're sure to return the indices in order (rotated)(L,R,P,Ps) ?

    return W, rotated_inds
end


"""
(L,R,P,P') => (P',P,R,L)
"""
function rotate_90clockwisen(W::ITensor; L=nothing, R=nothing, P=nothing, Ps=nothing)

    dim_rotV = dim(P)
    dim_rotP = isnothing(L) ? dim(R) : dim(L)

    irotL = Index(dim_rotV,"time,virtL")
    irotR = Index(dim_rotV,"time,virtR")
    irotP = Index(dim_rotP,"time,site")
    irotPs = Index(dim_rotP,"time,site")'

    if ndims(W) == 4 
        W = replaceinds(W, (L,R,P,Ps) ,(irotPs, irotP, irotR, irotL))
        W = permute(W, (irotL, irotR, irotP, irotPs))

    elseif ndims(W) == 3 && isnothing(L) # Wl 
        W = replaceinds(W, (R,P,Ps) ,(irotP, irotR, irotL))
        W = permute(W, (irotL, irotR, irotP))  # should be Ps but we already unprime here 

    elseif ndims(W) == 3 && isnothing(R) # Wr 
        W = replaceinds(W, (L,P,Ps) ,(irotP, irotR, irotL))
        W = permute(W, (irotL, irotR, irotP))
    else
        @error "Trying to rotate a Wtensor with $ndims(W) legs, not sure what to do"
    end

    rotated_inds = Dict(:L => irotL, :R => irotR, :P => irotP, :Ps => irotPs )

    return W, rotated_inds
end





""" Convention we stick to for all the following: indices for tMPO WWl before rotations are  L, R, P, P' """
function build_WW(tp::tmpo_params)
    eH = build_expH(tp)

    WWl,       iCwR, iCp, iCps = build_WWl(eH)
    WWr, iCwL,       iCp, iCps = build_WWr(eH)
    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    check_symmetry_itensor_mpo(WWc, iCwL, iCwR, iCp, iCps)

    return WWl, WWc, WWr
end

    

""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' 
"""
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


""" Builds folded unrotated left edge tensor of the network """ 
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

