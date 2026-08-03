""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution 
Rotated 90deg clockwise:  (L,R,P,P') => (P',P,L,R)
"""
struct FwtMPOBlocks
    Wl::ITensor
    Wc::ITensor
    Wr::ITensor
    Wl_im::ITensor
    Wc_im::ITensor
    Wr_im::ITensor
    tp::tMPOParams
    iL::Index
    iR::Index
    iP::Index
    iPs::Index

    function FwtMPOBlocks(Wl::ITensor,Wc::ITensor,Wr::ITensor, Wl_im::ITensor,Wc_im::ITensor,Wr_im::ITensor, tp::tMPOParams, iL::Index, iR::Index, iP::Index, iPs::Index)

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new( adapt(dttype,Wl), adapt(dttype,Wc), adapt(dttype, Wr), adapt(dttype,Wl_im), adapt(dttype,Wc_im), adapt(dttype, Wr_im), tp, iL, iR, iP, iPs)
    end
end

ITensorMPS.linkinds(b::FwtMPOBlocks) = (b.iL, b.iR)
ITensorMPS.siteinds(b::FwtMPOBlocks) = (b.iP, b.iPs)
ITensorMPS.siteind(b::FwtMPOBlocks) = b.iP


function FwtMPOBlocks(tp::tMPOParams; init_state=nothing)
    Wl, Wc, Wr, iL, iR, iP, iPs = make_fwtmpoblocks(tp)

    if !isnothing(init_state)
        @info "Setting tp.bl to $(init_state)"
        tp = tMPOParams(tp.mp; dt=tp.dt, dbeta=tp.dbeta, scheme=tp.scheme, nbeta=tp.nbeta, init_state=init_state)
    end

    Wl_im, Wc_im, Wr_im, iL_im, iR_im, iP_im, iPs_im = make_fwtmpoblocks(tp; build_imag=true)
    iminds = (iL_im, iR_im, iP_im, iPs_im)
    inds   = (iL,    iR,    iP,    iPs)

    Wl_im = replaceinds(Wl_im, iminds, inds)
    Wc_im = replaceinds(Wc_im, iminds, inds)
    Wr_im = replaceinds(Wr_im, iminds, inds)
    
    return FwtMPOBlocks(Wl, Wc, Wr, Wl_im, Wc_im, Wr_im, tp, iL, iR, iP, iPs)

end

function FwtMPOBlocks(eH::MPO; init_state)
    tp = tMPOParams(nothing; bl=init_state)
    Wl, Wc, Wr, iL, iR, iP, iPs = make_fwtmpoblocks(eH)
    return FwtMPOBlocks(Wl, Wc, Wr, Wl, Wc, Wr, tp, iL, iR, iP, iPs)
end

FwtMPOBlocks(scheme::ExpHRecipe, mp::ModelParams; dt::Number=0.1, init_state, kwargs...) =
    FwtMPOBlocks(tMPOParams(mp; dt, scheme, init_state); kwargs...)

""" Allow changing elements of FwtMPOBlocks """
function FwtMPOBlocks(b::FwtMPOBlocks; 
    Wl=b.Wl, Wc=b.Wc, Wr=b.Wr, Wl_im=b.Wl_im, Wc_im=b.Wc_im, Wr_im=b.Wr_im, tp=b.tp,
    iL=b.iL, iR=b.iR, iP=b.iP, iPs=b.iPs)
    return FwtMPOBlocks(Wl, Wc, Wr, Wl_im, Wc_im, Wr_im, tp, iL, iR, iP, iPs)
end

function make_fwtmpoblocks(tp::tMPOParams; build_imag::Bool=false)
    dt = build_imag ? tp.dbeta : tp.dt 
    make_fwtmpoblocks(build_Ut(tp; dt))
end

function make_fwtmpoblocks(eH::MPO; check_sym::Bool=true)

    @assert length(eH) == 3

    (Wl, Wc, Wr) = eH

    (ilP, icP, irP) = firstsiteinds(eH)
    (iLink1, iLink2) = linkinds(eH)

    if check_sym
        @info "Checking symmetry MPO tensor on physical(space) => bond(time) indices"
        check_symmetry_swap(Wc, icP, icP')
        @info "Checking symmetry MPO tensor on bond(space) => phys(time) indices"
        check_symmetry_swap(Wc, iLink1, iLink2)
    end

    # Rotated indices. With QNs the arrows matter: the two temporal *site* legs must be
    # opposite (as for any MPO), and so must the two temporal *links*. We take the arrows
    # from the legs as they are stored in the bulk tensor `Wc` - note that `icP'` (the
    # prime of the site index) carries the *same* arrow as `icP`, whereas the leg actually
    # sitting in `Wc` is its dagger, so we cannot sim() it directly.
    # `dag` is a no-op without QNs, so this reproduces the old indices exactly.
    time_P  = sim(iLink2, tags="Site,time")
    time_Ps = dag(time_P)'
    time_vL = sim(icP, tags="Link,time")
    time_vR = dag(sim(icP, tags="Link,time"))


    """  (L,R,P,P') => (P',P,L,R) """
    Wl = replaceinds(Wl, (iLink1,ilP,ilP'), (time_Ps, time_vL, time_vR))
    Wc = replaceinds(Wc, (iLink1,iLink2,icP,icP'), (time_Ps, time_P,time_vL, time_vR))
    Wr = replaceinds(Wr, (iLink2,irP,irP'), (time_P,time_vL, time_vR))


    return Wl, Wc, Wr, time_vL, time_vR, time_P, time_Ps
end


get_Ws(b::FwtMPOBlocks; imag::Bool=false) = imag ? (b.Wl_im, b.Wc_im, b.Wr_im) : (b.Wl, b.Wc, b.Wr)




Adapt.adapt_structure(to, b::FwtMPOBlocks) = FwtMPOBlocks(b;
    Wl=adapt(to, b.Wl), Wc=adapt(to, b.Wc), Wr=adapt(to, b.Wr),
    Wl_im=adapt(to, b.Wl_im), Wc_im=adapt(to, b.Wc_im), Wr_im=adapt(to, b.Wr_im),
    tp=adapt(to, b.tp))


function Base.show(io::IO, b::FwtMPOBlocks)
    println(io, "forward tMPO Blocks, type $(NDTensors.unwrap_array_type(b.Wc))")
    println(io, b.tp)
end