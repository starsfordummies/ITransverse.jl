""" Build folded tMPO with an extra site at the beginning for the initial state, which we specify as `init_tensor`
with a physical index `init_physidx`.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function folded_tMPO_in(b::FoldtMPOBlocks, ts::Vector{<:Index}; init_tensor::ITensor, init_physidx::Index, kwargs...) 

    dttype = NDTensors.unwrap_array_type(b.WWc)

    init_tensor = adapt(dttype, init_tensor)
    links_rho0 = uniqueinds(inds(init_tensor), init_physidx)

    # Just build an extended folded_tMPO and replace the first tensor with the initial state 

    tp_ext = tMPOParams(b.tp; nbeta = b.tp.nbeta+1)
    ww = folded_tMPO(FoldtMPOBlocks(b; tp=tp_ext), ts; kwargs...)

    init_tensor = init_tensor * delta(init_physidx, linkind(ww,1))
    init_tensor = init_tensor * delta(links_rho0[1], siteind(ww,1)')
    init_tensor = init_tensor * delta(links_rho0[2], siteind(ww,1))

    replace!(ww.data, ww[1] => init_tensor)

    return ww

end

""" Build folded tMPO with an extra site at the beginning for the initial state, which we specify as `init_tensor`
with a physical index `init_physidx`.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function fw_tMPO_in(b::FwtMPOBlocks, ts::Vector{<:Index}; init_tensor::ITensor, init_physidx::Index, kwargs...) 

    dttype = NDTensors.unwrap_array_type(b.Wc)

    init_tensor = adapt(dttype, init_tensor)
    links_psi0 = uniqueinds(inds(init_tensor), init_physidx)

    # Just build an extended folded_tMPO and replace the first tensor with the initial state 

    tp_ext = tMPOParams(b.tp; nbeta = b.tp.nbeta+1)
    ww = folded_tMPO(FwtMPOBlocks(b; tp=tp_ext), ts; kwargs...)

    init_tensor = init_tensor * delta(init_physidx, linkind(ww,1))
    init_tensor = init_tensor * delta(links_psi0[1], siteind(ww,1)')
    init_tensor = init_tensor * delta(links_psi0[2], siteind(ww,1))

    @show ww[1][1,1,1]
    ww.data[1] = init_tensor
    @show ww[1][1,1,1]

    return ww

end



""" Build folded tMPO with an extra site at the beginning for the initial state, which we specify as `init_tensor`
with a physical index `init_physidx`.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function tMPO_in(b, ts::Vector{<:Index}; init_tensor::ITensor, init_physidx::Index, kwargs...) 

    tp_ext = tMPOParams(b.tp; nbeta = b.tp.nbeta+1)

    ww, dttype = if b isa FwtMPOBlocks
        fw_tMPO(FwtMPOBlocks(b; tp=tp_ext), ts; kwargs...), NDTensors.unwrap_array_type(b.Wc)
    elseif b isa FoldtMPOBlocks
        folded_tMPO(FoldtMPOBlocks(b; tp=tp_ext), ts; kwargs...), NDTensors.unwrap_array_type(b.WWc)
    end

    init_tensor = adapt(dttype, init_tensor)
    links_psi0 = uniqueinds(inds(init_tensor), init_physidx)
    @assert length(links_psi0) == 2 

    # Just build an extended tMPO and replace the first tensor with the initial state 

    init_tensor = replaceind(init_tensor, init_physidx, linkind(ww,1))
    init_tensor = replaceind(init_tensor, links_psi0[1], links_psi0[2]')

    #@show ww[1][1,1,1]
    ww.data[1] = init_tensor
    #@show ww[1][1,1,1]


    return ww

end