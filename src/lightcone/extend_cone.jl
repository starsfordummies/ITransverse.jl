"""
One step of the light cone algorithm: takes left and right tMPS ll, rr,
the time MPO and the operator O
extends the left-right tMPS by building extended tMPOs 1 and O, applying them and and optimizing 
1) the overlap (L1|OR)  -> save new L
2) the overlap (LO|1R)  -> save new R  (in case non symmetric)

Returns the updated left-right tMPS 
"""
function extend_tmps_cone(ll::MPS, op_L::Vector{<:Number}, op_R::Vector{<:Number}, rr::MPS, 
    ts::Vector{<:Index}, b::FoldtMPOBlocks, truncp::TruncParams)

    @assert length(ts) == length(ll)+1 

    tmpo = folded_tMPO_R(b, ts; fold_op=op_R)

    psi_R = apply_extend(tmpo, rr)

    tmpo = folded_tMPO_L(b, ts; fold_op=op_L) 
    # =swapprime(folded_tMPO(tp, time_sites; fold_op=op_L), 0, 1, "Site")

    psi_L = apply_extend(tmpo, ll)

    #ll, rr, ents, ov = truncate_rsweep(psi_L,psi_R; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    ll, rr, ents = truncate_sweep(psi_L,psi_R, truncp)
    
    return ll, rr, ents

end


function extend_tmps_cone_LOR(ll::MPS, op_L::Vector{<:Number}, op_R::Vector{<:Number}, rr::MPS, 
    ts::Vector{<:Index}, b::FoldtMPOBlocks, truncp::TruncParams)

    @assert length(ts) == length(ll)+1 

    tmpoR = folded_tMPO_R(b, ts; fold_op=op_R)

    psi_R = apply_extend(tmpoR, rr)

    tmpo = folded_tMPO(b, ts; fold_op=op_L) 

    LO = applyns(tmpo, psi_R)

    #ll, rr, ents, ov = truncate_rsweep(psi_L,psi_R; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    _, rr, ents = truncate_sweep(LO,psi_R, truncp)
    
    return ll, rr, ents

end

