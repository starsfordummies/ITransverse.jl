"""
One step of the light cone algorithm: takes left and right tMPS ll, rr, 
the operator on the left and right columns, 
the (already extended) time sites and folded tMPO blocks.
Extends the left-right tMPS by building extended tMPOs Eleft and Eright, applying them and and optimizing 
the overlap (LEleft|ErightR). Returns truncated tMPS Lnew, Rnew and SVD entropies


Returns the updated left-right tMPS 
"""
function extend_tmps_cone(ll::MPS, op_L::Vector{<:Number}, op_R::Vector{<:Number}, rr::MPS, 
    ts::Vector{<:Index}, b::FoldtMPOBlocks, truncp::TruncParams)

    # We can extend by more than one timestep if the cone is narrow
    n_ext = length(ts) - length(ll)

    tmpo = folded_tMPO_ext(b, ts; LR="R", fold_op=op_R, n_ext)
    psi_R = applyn(tmpo, rr)

    tmpo = folded_tMPO_ext(b, ts; LR="L", fold_op=op_L, n_ext) 
    psi_L = applyns(tmpo, ll)

    ll, rr, ents = truncate_sweep(psi_L,psi_R, truncp)
    
    return ll, rr, ents

end
