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


function FwtMPOBlocks(tp::tMPOParams; build_imag::Bool=true, init_state=nothing, kwargs...)
    Wl, Wc, Wr, rot_inds = make_fwtmpoblocks(build_expH(tp; kwargs...))

    if !isnothing(init_state)
        @info "Setting tp.init_state to $(init_state)"
        tp = tMPOParams(tp; bl=init_state)
    end

    if build_imag
        Wl_im, Wc_im, Wr_im, rot_inds_im = make_fwtmpoblocks(build_expHim(tp; kwargs...))
        iminds = (rot_inds_im[:L], rot_inds_im[:R], rot_inds_im[:P], rot_inds_im[:Ps])
        inds =   (   rot_inds[:L],    rot_inds[:R],    rot_inds[:P],    rot_inds[:Ps])

        Wl_im = replaceinds(Wl_im, iminds, inds)
        Wc_im = replaceinds(Wc_im, iminds, inds)
        Wr_im = replaceinds(Wr_im, iminds, inds)
        
        return FwtMPOBlocks(Wl, Wc, Wr, Wl_im, Wc_im, Wr_im, tp, rot_inds)
    else
        return FwtMPOBlocks(Wl, Wc, Wr, Wl, Wc, Wr, tp, rot_inds)
    end

end

function FwtMPOBlocks(eH::MPO; init_state=nothing)
    tp = tMPOParams(nothing; bl=init_state)
    Wl, Wc, Wr, rot_inds = make_fwtmpoblocks(eH)
    return FwtMPOBlocks(Wl, Wc, Wr, Wl, Wc, Wr, tp, rot_inds)

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



    #time_P = Index(dim(iLink1),"Site,time")
    time_P = sim(iLink1, tags="Site,time")
    #time_vL = Index(dim(icP),"Link,time")
    time_vL = sim(icP, tags="Link,time")
    #time_vR = Index(dim(icP'),"Link,time")
    time_vR = sim(icP', tags="Link,time")


    """  (L,R,P,P') => (P',P,L,R) """
    Wl = replaceinds(Wl, (iLink1,ilP,ilP'), (time_P', time_vL, time_vR))
    Wc = replaceinds(Wc, (iLink1,iLink2,icP,icP'), (time_P', time_P,time_vL, time_vR))
    Wr = replaceinds(Wr, (iLink2,irP,irP'), (time_P,time_vL, time_vR))


    rot_inds = Dict(:Ps => time_P',:P => time_P, :L => time_vL, :R=> time_vR) 

    return Wl, Wc, Wr, rot_inds
end





Adapt.adapt_structure(to, b::FwtMPOBlocks) = FwtMPOBlocks(
    adapt(to, b.Wl), 
    adapt(to, b.Wc), 
    adapt(to, b.Wr),
    adapt(to, b.Wl_im), 
    adapt(to, b.Wc_im), 
    adapt(to, b.Wr_im),
    adapt(to, b.tp), 
    b.rot_inds
)
