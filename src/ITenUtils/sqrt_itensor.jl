using ITensors.Adapt: adapt
function Base.sqrt(a::ITensor)
    #data_type = mapreduce(NDTensors.unwrap_array_type, promote_type, a)
    data_type = promote_type(NDTensors.unwrap_array_type(a))
     #y = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, x), random_mps(s))

    if isdiag(a, 1e-10) # make back to diagonal and sqrt it
        sq_a = diag_itensor(sqrt.(array(diag(a))), inds(a))
    else
        a = NDTensors.cpu(a)
        sq_a = ITensor(sqrt(matrix(a)), inds(a))
        sq_a = adapt(data_type, sq_a)
    end

end
