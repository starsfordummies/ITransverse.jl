""" Vectorizes an MPO by "folding" (joining) its physical indices.
 Returns the corresponding MPS *and* the list of combiners used to join the indices, for later use.
 We assume standard (p,p') labelling and try to join indices as (phys, phys')
 """
function vectorize_mpo(w::MPO)

    sites_p =  siteinds(first, w, plev=0)
    sites_ps =  siteinds(first, w, plev=1)
    tensors = deepcopy(w.data)

    combiners = [combiner(sites_p[jj], sites_ps[jj], tags="fold, s=$(jj)") for jj in eachindex(w)]

    for jj in eachindex(w)
        tensors[jj] *= combiners[jj]
    end

    return MPS(tensors), combiners
end


""" Given a vectorized MPO (ie. an MPS) and the list of combiners used, unfolds the MPS converting it back to MPO """
function unvectorize_mpo(w::MPS, combiners)

    tensors = deepcopy(w.data)

    for jj in eachindex(w)
        tensors[jj] *= dag(combiners[jj])
    end

    return MPO(tensors)
end
