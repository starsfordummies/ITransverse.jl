
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
    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH)
    WWl,       iCwR, iCp, iCps = build_WWl(eH)
    WWr, iCwL,       iCp, iCps = build_WWr(eH)

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


""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution 
 * already rotated 90deg clockwise* - so the physical indices are "temporal" ones.
"""
struct FoldtMPOBlocks
    WWl::ITensor
    WWc::ITensor
    WWr::ITensor
    rho0::ITensor
    tp::tmpo_params
    rot_inds::Dict

    function FoldtMPOBlocks(tp::tmpo_params, init_state::Vector{<:Number} = tp.bl) 

        WWl, WWc, WWr = build_WW(tp::tmpo_params)

        R, P, Ps = inds(WWl)
        WWl, _ = rotate_90clockwise(WWl; R,P,Ps)
        L, R, P, Ps = inds(WWc)
        WWc, rotated_inds= rotate_90clockwise(WWc; L,R,P,Ps)
        L, P, Ps = inds(WWr)
        WWr, _ = rotate_90clockwise(WWr; L,P,Ps)

        # Match the inds of all three tensors FIXME do we want this? 
        # TODO check: rotated ints should be L, R, P, P' 
        iwl = (Index(1), inds(WWl)...) 
        WWl = replaceinds(WWl, iwl, inds(WWc))
        iwr = (ind(WWr,1), Index(1), ind(WWr,2),ind(WWr,3))
        WWr = replaceinds(WWr, iwr, inds(WWc))

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

        inds_ww = Dict() # TODO
        return new(WWl, WWc, WWr, rho0, tp, inds_ww)
    end

end




""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution rotated 90deg clockwise"""
struct FwtMPOBlocks
    Wl::ITensor
    Wc::ITensor
    Wr::ITensor
    tp::tmpo_params
    rot_inds::Dict

    function FwtMPOBlocks(tp::tmpo_params)
    
        eH = build_expH(tp)
    
        Wl, Wc, Wr = eH.data

        #space_pL = siteind(eH,1)
        ilP = siteind(eH,1)
        ilPs = ilP'

        icP = siteind(eH,2)
        icPs = icP'

        irP = siteind(eH,3)
        irPs = irP'
    
        (iL, iR) = linkinds(eH)

        check_symmetry_itensor_mpo(Wc, iL, iR, icP, icP')

        # rotate 90deg 

        Wl, rotated_inds = rotate_90clockwisen(Wl;      R=iL,P=ilP,Ps=ilPs)
        Wc, rotated_inds = rotate_90clockwisen(Wc; L=iL,R=iR,P=icP,Ps=icPs)
        Wr, rotated_inds = rotate_90clockwisen(Wr; L=iR,     P=irP,Ps=irPs)
    
        rot_inds = Dict() #TODO

        return new(Wl, Wc, Wr, tp, rot_inds)
    end

end
