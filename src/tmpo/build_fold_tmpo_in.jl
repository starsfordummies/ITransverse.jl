""" Build folded tMPO with an extra site at the beginning for the initial state, which we specify as `init_tensor`
with a physical index `init_physidx`.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function folded_tMPO_in(b::FoldtMPOBlocks, ts::Vector{<:Index}; init_tensor::ITensor, init_physidx::Index, kwargs...) 

    dttype = NDTensors.unwrap_array_type(b.WWc)

    init_tensor = adapt(dttype, init_tensor)
    links_rho0 = uniqueinds(inds(init_tensor), init_physidx)

    init_tensor = init_tensor * delta(init_physidx, dag(b.rot_inds[:P]))
    init_tensor = init_tensor * delta(links_rho0[1], dag(links_rho0[2])')

    # Just build an extended folded_tMPO and replace the first tensor with the initial state 

    tp_ext = tMPOParams(b.tp; nbeta = b.tp.nbeta+1)
    ww = folded_tMPO(FoldtMPOBlocks(b; tp=tp_ext), ts; kwargs...)

    replace!(ww.data, ww[1] => init_tensor)

    return ww

end
