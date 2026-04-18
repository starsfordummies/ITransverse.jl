"""
Vectorize an MPO by folding each (p, p') pair into a single site index.
 We assume standard (p,p') labelling and try to join indices as (phys, phys')
Returns the MPS and the combiners (needed to unvectorize).
"""
function vectorize_mpo(w::MPO)
    sites_p  = siteinds(first, w, plev=0)
    sites_ps = siteinds(first, w, plev=1)

    combiners = [combiner(sites_p[i], sites_ps[i]; tags="fold,s=$i")
                 for i in eachindex(w)]

    tensors = [w[i] * combiners[i] for i in eachindex(w)]

    return MPS(tensors), combiners
end

""" Given a vectorized MPO (ie. an MPS) and the list of combiners used, unfolds the MPS converting it back to MPO """
function unvectorize_mpo(w::MPS, combiners)

    tensors = [w[i] * dag(combiners[i]) for i in eachindex(w)]
    return MPO(tensors)
end


""" Returns an MPS with the (folded) MPO for a local operator `local_op` at `site_op` """
function vectorized_local_op(ss::Vector{<:Index}; local_op::String="Id", site_op=div(length(ss)+1,2))

    local_ops = [op("Id", s) for s in ss]
    local_ops[site_op] = op(local_op, ss[site_op])

    o_local_ops = MPO(local_ops)

    vo_local_ops, combiners = vectorize_mpo(o_local_ops)
end

function fidelity(psi::MPO, phi::MPO; match_inds::Bool=true)
    o1, _ = vectorize_mpo(psi)
    o2, _ = vectorize_mpo(phi)
    fidelity(o1, o2; match_inds)
end
