""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution 
Rotated 90deg clockwise:  (L,R,P,P') => (P',P,L,R)
"""
struct FwtMPOBlocks
    Wl::ITensor
    Wc::ITensor
    Wr::ITensor
    tp::tMPOParams
    rot_inds::Dict

    function FwtMPOBlocks(Wl::ITensor,Wc::ITensor,Wr::ITensor, tp::tMPOParams,rot_inds::Dict) 

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new( adapt(dttype,Wl), adapt(dttype,Wc), adapt(dttype, Wr), tp, rot_inds)
    end
end

function FwtMPOBlocks(tp::tMPOParams)
    
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
"""  (L,R,P,P') => (P',P,L,R) """
    Wl = replaceinds(Wl, (iLink1,ilP,ilPs), (time_P, time_vL, time_vR))
    Wc = replaceinds(Wc, (iLink1,iLink2,icP,icPs), (time_P', time_P,time_vL, time_vR))
    Wr = replaceinds(Wr, (iLink2,irP,irPs), (time_P,time_vL, time_vR))

    # Wl, rotated_inds = rotate_90clockwise(Wl;      R=iL,P=ilP,Ps=ilPs)
    # Wc, rotated_inds = rotate_90clockwise(Wc; L=iL,R=iR,P=icP,Ps=icPs)
    # Wr, rotated_inds = rotate_90clockwise(Wr; L=iR,     P=irP,Ps=irPs)

    rot_inds = Dict(:Ps => time_P',:P => time_P, :L => time_vL, :R=> time_vR) 

    return FwtMPOBlocks(Wl, Wc, Wr, tp, rot_inds)
end
