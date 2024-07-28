
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

    function FoldtMPOBlocks(WWl::ITensor,WWc::ITensor,WWr::ITensor,
        rho0::ITensor,tp::tmpo_params,rot_inds::Dict) 

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new( adapt(dttype,WWl), adapt(dttype,WWc), adapt(dttype, WWr), adapt(dttype, rho0), tp, rot_inds)
    end
end


function FoldtMPOBlocks(tp::tmpo_params, init_state::ITensor = tp.bl) 

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


    # Handling of initial state 
    # We can accept either an initial state or initial folded state (DM)
    iP = phys_ind(init_state)
    physdim_init = dim(iP)

    @assert ndims(init_state) == 1 || ndims(init_state) == 3

    lrinds = uniqueinds(inds(init_state), iP)
    if !isempty(lrinds)
        @assert dim(lrinds[1]) == dim(lrinds[2])  # translational invariance 
    end

    # Do we need to fold ? 
    if physdim_init == dim(P)
        rho0 = init_state * delta(iP, Index(dim(P),"virt,time,rho0"))
        if !isempty(lrinds)
            iP_rho0 = Index(dim(rho0,1),"rho0,time,phys")
            rho0 = replaceinds(rho0, lrinds, (iP_rho0, iP_rho0'))
        end
    elseif physdim_init == dim(P) ÷ 2
        rho0 = (init_state) * (init_state') 
        comb_phys = combiner(iP, iP')
        rho0 *= comb_phys
        
        if !isempty(lrinds)
            ciileft = combiner(lrinds[1], lrinds[1]') 
            rho0 *= ciileft
            ciiright = combiner(lrinds[2], lrinds[2]') 
            rho0 *= ciiright
            iP_rho0 = Index(dim(combinedind(ciileft)),"rho0,time,phys")
            @show iP_rho0
            @show ciileft
            rho0 = replaceinds(rho0, (combinedind(ciileft), combinedind(ciiright)), (iP_rho0, iP_rho0'))
        end
        rho0 *= delta(combinedind(comb_phys), Index(dim(P), "virt,time,rho0"))
    else
        @error "Dimension of init_state is $(physdim_init) vs linkdim $(dim(P))"
    end

    inds_ww = Dict() # TODO
    return FoldtMPOBlocks(WWl, WWc, WWr, rho0, tp, inds_ww)
end

""" Allow changing elements of FoldtMPOBlocks """
function FoldtMPOBlocks(b::FoldtMPOBlocks; 
    WWl=b.WWl, WWc=b.WWc, WWr=b.WWr, rho0=b.rho0, tp=b.tp, rot_inds=b.rot_inds)
    return FoldtMPOBlocks(WWl, WWc, WWr, rho0, tp, rot_inds)
end

