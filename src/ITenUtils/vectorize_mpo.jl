""" Vectorizes an MPO by "folding" (joining) its physical indices.
 Returns the corresponding MPS *and* the list of combiners used to join the indices, for later use"""
function vectorize_mpo(w::MPO)

    ss = siteinds(w)
    tensors = deepcopy(w.data)

    combiners = [combiner(ss[jj], tags="fold, s=$(jj)") for jj in eachindex(w)]

    for jj in eachindex(w)
        tensors[jj] *= combiners[jj]
    end

    return MPS(tensors), combiners
end

""" Vectorizes an MPO by "folding" (joining) its physical indices, 
inplace version which likely destroys the starting MPO to save resources.
 Returns the corresponding MPS *and* the list of combiners used to join the indices, for later use"""
function vectorize_mpo!(w::MPO)

    ss = siteinds(w)

    combiners = [combiner(ss[jj], tags="fold, s=$(jj)") for jj in eachindex(w)]

    for jj in eachindex(w)
        w.data[jj] *= combiners[jj]
    end

    return MPS(w.data), combiners
end


""" Given a vectorized MPO (ie. an MPS) and the list of combiners used, unfolds the MPS converting it back to MPO """
function unvectorize_mpo(w::MPS, combiners)

    tensors = deepcopy(w.data)

    for jj in eachindex(w)
        tensors[jj] *= dag(combiners[jj])
    end

    return MPS(tensors), combiners
end

""" Same as unvectorize_mpo but inplace destroying the input MPS """
function unvectorize_mpo!(w::MPS, combiners)

    for jj in eachindex(w)
        w.data[jj] *= dag(combiners[jj])
    end

    return MPS(w.data), combiners
end
