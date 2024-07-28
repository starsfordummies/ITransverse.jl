""" Basic building blocks for the folded tMPS/tMPO, folded tensors of time evolution rotated 90deg clockwise"""
struct FwtMPOBlocks
    Wl::ITensor
    Wc::ITensor
    Wr::ITensor
    tp::tmpo_params
    rot_inds::Dict

  
    function FwtMPOBlocks(Wl::ITensor,Wc::ITensor,Wr::ITensor, tp::tmpo_params,rot_inds::Dict) 

        # The data type of the bottom-left term in tp dictates whether the *full* thing will lie on GPU
        dttype = NDTensors.unwrap_array_type(tp.bl)

        new( adapt(dttype,Wl), adapt(dttype,Wc), adapt(dttype, Wr), tp, rot_inds)
    end
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
