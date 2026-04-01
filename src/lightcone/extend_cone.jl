"""
One step of the light cone algorithm: takes left and right tMPS ll, rr, 
the operator on the left and right columns, 
the (already extended) time sites and folded tMPO blocks.
"""
function extend_tmps_cone_new(ll::MPS, op_L::Vector{<:Number}, op_R::Vector{<:Number}, rr::MPS, 
    ts::Vector{<:Index}, b::FoldtMPOBlocks; kwargs...)

    # We can extend by more than one timestep if the cone is narrow
    n_ext = length(ts) - length(ll)

    tmpoL = folded_tMPO_ext(b, ts; LR=:left, fold_op=op_L, n_ext) 
    tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=op_R, n_ext)
   
    ll, rr, ents = tlrapply(ll, tmpoL, tmpoR, rr; kwargs...)
    
    #@show overlap_noconj(ll,rr)
    return ll, rr, ents

end