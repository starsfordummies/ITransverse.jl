
""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution 
 *already rotated 90deg clockwise* - so the physical indices are "temporal" ones.
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
    iL::Index
    iR::Index
    iP::Index
    iPs::Index

    function FoldtMPOBlocks(WWl::ITensor,WWc::ITensor,WWr::ITensor, WWl_im::ITensor,WWc_im::ITensor,WWr_im::ITensor,
        rho0::ITensor,tp::tMPOParams, iL::Index, iR::Index, iP::Index, iPs::Index)

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new(adapt(dttype,WWl), adapt(dttype,WWc), adapt(dttype, WWr), 
            adapt(dttype,WWl_im), adapt(dttype,WWc_im), adapt(dttype, WWr_im), 
            adapt(dttype, rho0), tp, iL, iR, iP, iPs)
    end
end

ITensorMPS.linkinds(b::FoldtMPOBlocks) = (b.iL, b.iR)
ITensorMPS.siteinds(b::FoldtMPOBlocks) = (b.iP, b.iPs)
ITensorMPS.siteind(b::FoldtMPOBlocks) = b.iP



""" Allow changing elements of FoldtMPOBlocks """
function FoldtMPOBlocks(b::FoldtMPOBlocks; 
    WWl=b.WWl, WWc=b.WWc, WWr=b.WWr, WWl_im=b.WWl_im, WWc_im=b.WWc_im, WWr_im=b.WWr_im, rho0=b.rho0, tp=b.tp,
    iL=b.iL, iR=b.iR, iP=b.iP, iPs=b.iPs)
    return FoldtMPOBlocks(WWl, WWc, WWr, WWl_im, WWc_im, WWr_im, rho0, tp, iL, iR, iP, iPs)
end



""" Builds FoldtMPOBlocks tensors making the rotated+folded tMPO (L,R,P,P') => (P',P,L,R)
from either tMPOParameters or directly from an MPO of U=exp(iHt) defined on spatial links """
function FoldtMPOBlocks(x::Union{tMPOParams, MPO}; init_state=nothing, check_sym::Bool=true)

    WWl, WWc, WWr, (link1, link2, P, Ps) = build_WW(x)
    time_P = Index(dim(link1), "Site,time")
    time_L = Index(dim(P), "Link,time")
    time_R = Index(dim(Ps), "Link,time")

    if check_sym
        symP = check_symmetry_swap(WWc, P, Ps; verbose=false)
        if ismissing(symP)
            @info "Symmetry checks skipped (QN tensor)"
        elseif symP
            @info "MPO tensor symmetric in physical(space)  [=bond(time)] indices"
        else
            @warn "MPO tensor *not* symmetric in physical(space)  [=bond(time)] indices"
        end

        symL = check_symmetry_swap(WWc, link1, link2; verbose=false)
        if ismissing(symL)
            # already reported above
        elseif symL
            @info "MPO tensor symmetric in bond(space) [=phys(time)]  indices"
        else
            @warn "MPO tensor *not* symmetric in bond(space) [=phys(time)] indices"
        end
    end

    unrotated_inds = (link1, link2, P, Ps)
    rotated_inds = (time_P', time_P, time_L, time_R)
    WWl = replaceinds(WWl, unrotated_inds, rotated_inds)
    WWc = replaceinds(WWc, unrotated_inds, rotated_inds)
    WWr = replaceinds(WWr, unrotated_inds, rotated_inds)

  
    tp, WWl_im, WWc_im, WWr_im = if x isa MPO 

        if isnothing(init_state)
            error("Need to specify initial state if we don't pass tp")
        else
            init_state = to_boundary(init_state)
        end
        # If the input is an MPO, we don't build anyting else, just put placeholders in tp 
        phys_site = siteind(x,2)
        mp = NoParams(phys_site)
        tp = tMPOParams(NaN, NaN, mp, Murg(), 0, init_state)
        tp, WWl, WWc, WWr

    else # x isa tMPOParams

        if isnothing(init_state)
            init_state = x.bl
        end

        tp = tMPOParams(x.dt, x.dbeta, x.mp, x.scheme, x.nbeta, to_boundary(init_state))

        
        WWl_im, WWc_im, WWr_im, unrotated_inds = build_WW(tp; build_imag=true)

        WWl_im = replaceinds(WWl_im, unrotated_inds, rotated_inds)
        WWc_im = replaceinds(WWc_im, unrotated_inds, rotated_inds)
        WWr_im = replaceinds(WWr_im, unrotated_inds, rotated_inds)

        tp, WWl_im, WWc_im, WWr_im

    end

    # Fold the initial state (if it isn't folded/vectorized already).
    # Non-product initial states keep their (doubled) bond legs, see `boundary_tensor`.
    rho0 = fold_boundary(tp.bl; folded_dim=dim(P), tags="Site,rho0")

    return FoldtMPOBlocks(WWl, WWc, WWr, WWl_im, WWc_im, WWr_im, rho0, tp, time_L, time_R, time_P, time_P')
end


get_Ws(b::FoldtMPOBlocks; imag::Bool=false) = imag ? (b.WWl_im, b.WWc_im, b.WWr_im) : (b.WWl, b.WWc, b.WWr)


Adapt.adapt_structure(to, b::FoldtMPOBlocks) = FoldtMPOBlocks(b;
    WWl=adapt(to, b.WWl),    WWc=adapt(to, b.WWc),    WWr=adapt(to, b.WWr),
    WWl_im=adapt(to, b.WWl_im), WWc_im=adapt(to, b.WWc_im), WWr_im=adapt(to, b.WWr_im),
    rho0=adapt(to, b.rho0), tp=adapt(to, b.tp))


function Base.show(io::IO, b::FoldtMPOBlocks)
    println(io, "[[*Folded*]] tMPO Blocks, type $(NDTensors.unwrap_array_type(b.WWc))")
    println(io, b.tp)
end