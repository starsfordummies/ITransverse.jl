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

function FwtMPOBlocks(tp::tMPOParams; build_imag::Bool=true)
     # TODO: build_imag iff nbeta != 0 ? 

    eH = build_expH(tp)

    Wl, Wc, Wr = eH.data

    #space_pL = siteind(eH,1)
    ilP = siteind(eH,1)
    ilPs = ilP'

    icP = siteind(eH,2)
    icPs = icP'

    irP = siteind(eH,3)
    irPs = irP'

    (iLink1, iLink2) = linkinds(eH)

    check_symmetry_itensor_mpo(Wc, iLink1, iLink2, icP, icP')

    # rotate 90deg 

    time_P = Index(dim(iLink1),"Site,time")
    time_vL = Index(dim(icP),"Link,time")
    time_vR = Index(dim(icPs),"Link,time")

    rot_inds = Dict(:Ps => time_P',:P => time_P, :L => time_vL, :R=> time_vR) 


    """  (L,R,P,P') => (P',P,L,R) """
    Wl = replaceinds(Wl, (iLink1,ilP,ilPs), (time_P, time_vL, time_vR))
    Wc = replaceinds(Wc, (iLink1,iLink2,icP,icPs), (time_P', time_P,time_vL, time_vR))
    Wr = replaceinds(Wr, (iLink2,irP,irPs), (time_P,time_vL, time_vR))


    #######################
    #### Imaginary time 
    #######################

    Wl_im = Wl 
    Wc_im = Wc 
    Wr_im = Wr

    if build_imag
        eHim = build_expHim(tp)

        Wl_im, Wc_im, Wr_im = eHim.data

        #space_pL = siteind(eH,1)
        ilP = siteind(eHim,1)
        ilPs = ilP'

        icP = siteind(eHim,2)
        icPs = icP'

        irP = siteind(eHim,3)
        irPs = irP'

        (iLink1, iLink2) = linkinds(eHim)

        Wl_im = replaceinds(Wl_im, (iLink1,ilP,ilPs), (time_P, time_vL, time_vR))
        Wc_im = replaceinds(Wc_im, (iLink1,iLink2,icP,icPs), (time_P', time_P,time_vL, time_vR))
        Wr_im = replaceinds(Wr_im, (iLink2,irP,irPs), (time_P,time_vL, time_vR))
    end
    
    return FwtMPOBlocks(Wl, Wc, Wr, Wl_im, Wc_im, Wr_im, tp, rot_inds)
end
