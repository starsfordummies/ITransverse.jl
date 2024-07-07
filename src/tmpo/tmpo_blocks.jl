
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

end


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
    return FoldtMPOBlocks(WWl, WWc, WWr, rho0, tp, inds_ww)
end



""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution rotated 90deg clockwise"""
struct FwtMPOBlocks
    Wl::ITensor
    Wc::ITensor
    Wr::ITensor
    tp::tmpo_params
    rot_inds::Dict
end

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

    return FwtMPOBlocks(Wl, Wc, Wr, tp, rot_inds)
end
