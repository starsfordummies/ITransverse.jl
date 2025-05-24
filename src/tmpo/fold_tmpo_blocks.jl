
""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution 
 * already rotated 90deg clockwise* - so the physical indices are "temporal" ones.
"""

struct FoldtMPOBlocks
    WWl::ITensor
    WWc::ITensor
    WWr::ITensor
    WWl_im::ITensor
    WWc_im::ITensor
    WWr_im::ITensor
    rho0::ITensor
    tp::tMPOParams
    rot_inds::Dict

    function FoldtMPOBlocks(WWl::ITensor,WWc::ITensor,WWr::ITensor, WWl_im::ITensor,WWc_im::ITensor,WWr_im::ITensor,
        rho0::ITensor,tp::tMPOParams,rot_inds::Dict) 

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new(adapt(dttype,WWl), adapt(dttype,WWc), adapt(dttype, WWr), 
            adapt(dttype,WWl_im), adapt(dttype,WWc_im), adapt(dttype, WWr_im), 
            adapt(dttype, rho0), tp, rot_inds)
    end
end


""" Allow changing elements of FoldtMPOBlocks """
function FoldtMPOBlocks(b::FoldtMPOBlocks; 
    WWl=b.WWl, WWc=b.WWc, WWr=b.WWr, WWl_im=b.WWl_im, WWc_im=b.WWc_im, WWr_im=b.WWr_im, rho0=b.rho0, tp=b.tp, rot_inds=b.rot_inds)
    return FoldtMPOBlocks(WWl, WWc, WWr, WWl_im, WWc_im, WWr_im, rho0, tp, rot_inds)
end



""" Starting from tMPOParams, builds the tensors making the rotated+folded tMPO
(L,R,P,P') => (P',P,L,R)
"""
function _FoldtMPOBlocks(tp::tMPOParams, init_state = tp.bl; check_sym::Bool=true) 

    WWl, WWc, WWr,  (L, R, P, Ps) = build_WW(tp::tMPOParams)
    
    init_state = to_itensor(init_state, "bl")

    time_P = Index(dim(L), "Site,time")
    time_L = Index(dim(P), "Link,time")
    time_R = Index(dim(Ps), "Link,time")
   
    WWl = replaceinds(WWl, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWc = replaceinds(WWc, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWr = replaceinds(WWr, (L,R,P,Ps),(time_P',time_P,time_L, time_R))

    # Same indices on WW and WW_im 

    tpim = tMPOParams(tp; dt = -im*tp.dt)
    WWl_im, WWc_im, WWr_im,  (L, R, P, Ps) = build_WW(tpim::tMPOParams)
   
    WWl_im = replaceinds(WWl_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWc_im = replaceinds(WWc_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWr_im = replaceinds(WWr_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))

   
    # Handling of initial state 
    # We can accept either an initial state or initial folded state (DM)
    iP = phys_ind(init_state)
    physdim_init = dim(iP)

    @assert ndims(init_state) == 1 || ndims(init_state) == 3

    lrinds = uniqueinds(inds(init_state), iP)
    if !isempty(lrinds)
        @assert dim(lrinds[1]) == dim(lrinds[2])  # translational invariance 
    end

    # Check if we need to fold 
    if physdim_init == dim(P)
        rho0 = init_state * delta(iP, Index(dim(P),"Link,time,rho0"))
        if !isempty(lrinds)
            iP_rho0 = Index(dim(rho0,1),"rho0,time,Site")
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
            iP_rho0 = Index(dim(combinedind(ciileft)),"rho0,time,Site")
            #@show iP_rho0
            #@show ciileft
            rho0 = replaceinds(rho0, (combinedind(ciileft), combinedind(ciiright)), (iP_rho0, iP_rho0'))
        end
        rho0 *= delta(combinedind(comb_phys), Index(dim(P), "Link,time,rho0"))
    else
        @error "Dimension of init_state is $(physdim_init) vs linkdim $(dim(P))"
    end

    inds_ww = Dict(:Ps => time_P',:P => time_P, :L => time_L, :R=> time_R) # TODO
    return FoldtMPOBlocks(WWl, WWc, WWr, WWl_im, WWc_im, WWr_im, rho0, tp, inds_ww)
end

""" Allow to pass a spatial MPO for U(t) and other tMPO params as input and build everything else from that """ 
function _FoldtMPOBlocks(eH_space_mpo::MPO, init_state; check_sym::Bool=true) 

    init_state = to_itensor(init_state, "bl")

    #TODO Done like this, we don't support beta yet 

    phys_site = siteind(eH_space_mpo,2)

    WWl, WWc, WWr,  (L, R, P, Ps) = build_WW(eH_space_mpo)

    time_P = Index(dim(L), "Site,time")
    time_L = Index(dim(P), "Link,time")
    time_R = Index(dim(Ps), "Link,time")
   
    WWl = replaceinds(WWl, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWc = replaceinds(WWc, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWr = replaceinds(WWr, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
   
    # Handling of initial state 
    # We can accept either an initial state or initial folded state (DM)
    iP = phys_ind(init_state)
    physdim_init = dim(iP)

    @assert ndims(init_state) == 1 || ndims(init_state) == 3

    lrinds = uniqueinds(inds(init_state), iP)
    if !isempty(lrinds)
        @assert dim(lrinds[1]) == dim(lrinds[2])  # translational invariance 
    end

    # Check if we need to fold 
    if physdim_init == dim(P)
        rho0 = init_state * delta(iP, Index(dim(P),"Link,time,rho0"))
        if !isempty(lrinds)
            iP_rho0 = Index(dim(rho0,1),"rho0,time,Site")
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
            iP_rho0 = Index(dim(combinedind(ciileft)),"rho0,time,Site")
            #@show iP_rho0
            #@show ciileft
            rho0 = replaceinds(rho0, (combinedind(ciileft), combinedind(ciiright)), (iP_rho0, iP_rho0'))
        end
        rho0 *= delta(combinedind(comb_phys), Index(dim(P), "Link,time,rho0"))
    else
        @error "Dimension of init_state is $(physdim_init) vs linkdim $(dim(P))"
    end

    mp = NoParams(phys_site)
    tp = tMPOParams(NaN, nothing, mp, 0, init_state)
    inds_ww = Dict(:Ps => time_P',:P => time_P, :L => time_L, :R=> time_R)
    return FoldtMPOBlocks(WWl, WWc, WWr, WWl, WWc, WWr, rho0, tp, inds_ww)
end

""" Builds FoldtMPOBlocks tensors making the rotated+folded tMPO (L,R,P,P') => (P',P,L,R)
from either tMPOParameters or directly from an MPO of U=exp(iHt) defined on spatial links """
function FoldtMPOBlocks(x::Union{tMPOParams, MPO}, init_state=nothing; check_sym::Bool=true)
    # --- Step 1: Build MPO blocks and indices ---

    WWl, WWc, WWr, (L, R, P, Ps) = build_WW(x)
    time_P = Index(dim(L), "Site,time")
    time_L = Index(dim(P), "Link,time")
    time_R = Index(dim(Ps), "Link,time")

    if check_sym
        @info "Checking symmetry MPO tensor on physical(space) => bond(time) indices"
        check_symmetry_swap(WWc, P, Ps)
        @info "Checking symmetry MPO tensor on bond(space) => phys(time) indices"
        check_symmetry_swap(WWc, L, R)
    end

    WWl = replaceinds(WWl, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWc = replaceinds(WWc, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
    WWr = replaceinds(WWr, (L,R,P,Ps),(time_P',time_P,time_L, time_R))

  

    WWl_im, WWc_im, WWr_im = WWl, WWc, WWr

    tp = if x isa tMPOParams
        # tMPOParams mode
        # Build -im*dt version
        tpim = tMPOParams(x; dt = -im*x.dt)
        WWl_im, WWc_im, WWr_im, (L, R, P, Ps) = build_WW(tpim)

        WWl_im = replaceinds(WWl_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
        WWc_im = replaceinds(WWc_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))
        WWr_im = replaceinds(WWr_im, (L,R,P,Ps),(time_P',time_P,time_L, time_R))

        if isnothing(init_state)
            init_state = x.bl
        end
        tp = x
    else # x isa MPO 

        if isnothing(init_state)
            @error "Need to specify initial state if we don't pass tp"
        else
            init_state = to_itensor(init_state, "bl")
        end

        phys_site = siteind(x,2)
        mp = NoParams(phys_site)
        tp = tMPOParams(NaN, nothing, mp, 0, init_state)
    end



    iP = phys_ind(init_state)
    physdim_init = dim(iP)
    @assert ndims(init_state) == 1 || ndims(init_state) == 3

    lrinds = uniqueinds(inds(init_state), iP)
    if !isempty(lrinds)
        @assert dim(lrinds[1]) == dim(lrinds[2])
    end

    # Folding logic
    if physdim_init == dim(P)
        rho0 = init_state * delta(iP, Index(dim(P),"Link,time,rho0"))
        if !isempty(lrinds)
            iP_rho0 = Index(dim(rho0,1),"rho0,time,Site")
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
            iP_rho0 = Index(dim(combinedind(ciileft)),"rho0,time,Site")
            rho0 = replaceinds(rho0, (combinedind(ciileft), combinedind(ciiright)), (iP_rho0, iP_rho0'))
        end
        rho0 *= delta(combinedind(comb_phys), Index(dim(P), "Link,time,rho0"))
    else
        @error "Dimension of init_state is $(physdim_init) vs linkdim $(dim(P))"
    end

    inds_ww = Dict(:Ps => time_P', :P => time_P, :L => time_L, :R=> time_R)

    return FoldtMPOBlocks(WWl, WWc, WWr, WWl_im, WWc_im, WWr_im, rho0, tp, inds_ww)
end

