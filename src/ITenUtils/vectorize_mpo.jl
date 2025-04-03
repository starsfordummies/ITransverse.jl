
function vectorize_mpo(w::MPO)

    ss = siteinds(w)
    tensors = deepcopy(w.data)

    combiners = [combiner(ss[jj], tags="fold, s=$(jj)") for jj in eachindex(w)]

    for jj in eachindex(w)
        tensors[jj] *= combiners[jj]
    end

    return MPS(tensors), combiners
end


function vectorize_mpo!(w::MPO)

    ss = siteinds(w)

    combiners = [combiner(ss[jj], tags="fold, s=$(jj)") for jj in eachindex(w)]

    for jj in eachindex(w)
        w.data[jj] *= combiners[jj]
    end

    return MPS(w.data), combiners
end


function unvectorize_mpo(w::MPS, combiners)

    tensors = deepcopy(w.data)

    for jj in eachindex(w)
        tensors[jj] *= dag(combiners[jj])
    end

    return MPS(tensors), combiners
end


function unvectorize_mpo!(w::MPS, combiners)

    for jj in eachindex(w)
        w.data[jj] *= dag(combiners[jj])
    end

    return MPS(w.data), combiners
end
