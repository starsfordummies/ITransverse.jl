
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


""" This is actually likely in the wrong direction - use the other function """ 
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
Given an MPO tensor `W` and the relevant indices,
We rotate our space vectors to the *right* by 90°, ie 

(L,R,P,P') => (P',P,L,R)
```

^                  ^
|       |p'        |     |L
t    L--o--R   =>  x  p--o--p'
|       |p         |     |R
|                  |
----x--->          -----t--->  

```
"""
function _2rotate_90clockwise(W::ITensor; L=nothing, R=nothing, P=nothing, Ps=nothing)

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


""" Try to do it right this time
(L,R,P,P') => (P',P,L,R)
 """
function _3rotate_90clockwise(W::ITensor; L=nothing, R=nothing, P=nothing, Ps=nothing)

    dim_rotV = dim(P)
    dim_rotP = isnothing(L) ? dim(R) : dim(L)

    irotL = Index(dim_rotV,"time,virt,L")
    irotR = Index(dim_rotV,"time,virt,R")
    irotP = Index(dim_rotP,"time,site")
    irotPs = Index(dim_rotP,"time,site")'

    if ndims(W) == 4 
        W = replaceinds(W, (L,R,P,Ps) ,(irotPs, irotP, irotL, irotR))
        W = permute(W, (irotL, irotR, irotP, irotPs))

    # For rank 3 tensors we think of them as MPS tensors and always get them a physical leg
    elseif ndims(W) == 3 && isnothing(L) # Wl 
        W = replaceinds(W, (R,P,Ps) ,(irotP, irotL, irotR))
        W = permute(W, (irotL, irotR, irotP))  # should be Ps but we already unprime here 

    elseif ndims(W) == 3 && isnothing(R) # Wr 
        W = replaceinds(W, (L,P,Ps) ,(irotP, irotL, irotR))
        W = permute(W, (irotL, irotR, irotP))
    else
        @error "Trying to rotate a Wtensor with $ndims(W) legs, not sure what to do"
    end

    rotated_inds = Dict(:L => irotL, :R => irotR, :P => irotP, :Ps => irotPs )

    return W, rotated_inds
end



""" Convention we stick to for all the following: indices for tMPO WWl before rotations are  L, R, P, P'
Builds Folded and UNROTATED tensors, just W * Wdag and joined indices """
function build_WW(eH::MPO)

    space_phys = Index(maxlinkdim(eH)^2, "Site,space")
    space_vleft = Index(dim(siteind(eH,2))^2, "Link,space")
    space_vright = Index(dim(siteind(eH,2))^2, "Link,space")

    WWl,       iCwR, iCp, iCps = build_WWl(eH)
    WWl = replaceinds(WWl, (iCwR, iCp, iCps),  (space_vright, space_phys, space_phys'))
    WWr, iCwL,       iCp, iCps = build_WWr(eH)
    WWr = replaceinds(WWr, (iCwL, iCp, iCps),  (space_vleft, space_phys, space_phys'))
    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    check_symmetry_itensor_mpo(WWc, iCwL, iCwR, iCp, iCps)
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

    #_, _, Wr = eH_space.data
    Wr = eH_space[end]

    space_p = siteind(eH_space,3)

    vL = linkinds(eH_space)[2]

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

    #Wl, _, _ = eH_space.data
    Wl = eH_space[1]

    space_p = siteind(eH_space,1)

    vR = linkinds(eH_space)[1]

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

