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
    rot_inds::Dict

    function FwtMPOBlocks(Wl::ITensor,Wc::ITensor,Wr::ITensor, Wl_im::ITensor,Wc_im::ITensor,Wr_im::ITensor, tp::tMPOParams, rot_inds::Dict) 

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new( adapt(dttype,Wl), adapt(dttype,Wc), adapt(dttype, Wr), adapt(dttype,Wl_im), adapt(dttype,Wc_im), adapt(dttype, Wr_im), tp, rot_inds)
    end
end


function FwtMPOBlocks(tp::tMPOParams; kwargs...)
     # TODO: build_imag iff nbeta != 0 ? 

    eH = build_expH(tp)
    FwtMPOBlocks(eH; tp, kwargs...)
end


function FwtMPOBlocks(eH::MPO; tp=nothing, init_state = nothing, build_imag::Bool=true, check_sym::Bool=true)

    @assert length(eH) == 3

    if isnothing(init_state)
        @info "No init state specified, defaulting to tp.bl"
        @info "tp.bl = $(tp.bl)"
        init_state = to_itensor(tp.bl, "bl")
    end

    init_state = to_itensor(init_state, "bl")

    if isnothing(tp)
        phys_site = siteind(eH,2)
        mp = NoParams(phys_site)
        tp = tMPOParams(NaN, nothing, mp, 0, init_state)
        build_imag = false
    end

    # Check whether the initial state makes sense 
    @assert dim(init_state) == dim(siteind(eH,2))

    @info "Updating init_state in tMPOParams to $(init_state)"
    tp = tMPOParams(tp; bl=to_itensor(init_state, "bl"))
 

    (Wl, Wc, Wr) = eH

    (ilP, icP, irP) = firstsiteinds(eH)
    (iLink1, iLink2) = linkinds(eH)

    if check_sym
        @info "Checking symmetry MPO tensor on physical(space) => bond(time) indices"
        check_symmetry_swap(Wc, icP, icP')
        @info "Checking symmetry MPO tensor on bond(space) => phys(time) indices"
        check_symmetry_swap(Wc, iLink1, iLink2)
    end


    time_P = Index(dim(iLink1),"Site,time")
    time_vL = Index(dim(icP),"Link,time")
    time_vR = Index(dim(icP'),"Link,time")


    """  (L,R,P,P') => (P',P,L,R) """
    Wl = replaceinds(Wl, (iLink1,ilP,ilP'), (time_P, time_vL, time_vR))
    Wc = replaceinds(Wc, (iLink1,iLink2,icP,icP'), (time_P', time_P,time_vL, time_vR))
    Wr = replaceinds(Wr, (iLink2,irP,irP'), (time_P,time_vL, time_vR))


    #######################
    #### Imaginary time 
    #######################

    Wl_im = Wl 
    Wc_im = Wc 
    Wr_im = Wr

    if build_imag
        eHim = build_expHim(tp)

        Wl_im, Wc_im, Wr_im = eHim.data

        (ilP, icP, irP) = firstsiteinds(eHim)

        (iLink1, iLink2) = linkinds(eHim)

        Wl_im = replaceinds(Wl_im, (iLink1,ilP,ilP'), (time_P, time_vL, time_vR))
        Wc_im = replaceinds(Wc_im, (iLink1,iLink2,icP,icP'), (time_P', time_P,time_vL, time_vR))
        Wr_im = replaceinds(Wr_im, (iLink2,irP,irP'), (time_P,time_vL, time_vR))
    end


    rot_inds = Dict(:Ps => time_P',:P => time_P, :L => time_vL, :R=> time_vR) 

    return FwtMPOBlocks(Wl, Wc, Wr, Wl_im, Wc_im, Wr_im, tp, rot_inds)
end
