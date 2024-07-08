using ITensors.Adapt: adapt
# function Base.sqrt(a::ITensor)
#     #data_type = mapreduce(NDTensors.unwrap_array_type, promote_type, a)
#     data_type = promote_type(NDTensors.unwrap_array_type(a))
#      #y = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, x), random_mps(s))

#     if isdiag(a, 1e-10) # make back to diagonal and sqrt it
#         sq_a = diag_itensor(sqrt.(array(diag(a))), inds(a))
#     else
#         a = NDTensors.cpu(a)
#         sq_a = ITensor(sqrt(matrix(a)), inds(a))
#         sq_a = adapt(data_type, sq_a)
#     end

# end

""" Sqrt for a square ITensor"""
function Base.sqrt(a::ITensor, is::Tuple{<:Index, <:Index}=inds(a))

    # we want to sqrt square matrices
    # We don't want to do any funny reshaping here 
    @assert ndims(a) == 2 
    @assert dim(a,1) == dim(a,2) == dim(is[1]) == dim(is[2])

    
    #@assert dim(is[1]) == dim(a,1)
    #@assert dim(is[2]) == dim(a,2)

    #data_type = mapreduce(NDTensors.unwrap_array_type, promote_type, a)
    data_type = promote_type(NDTensors.unwrap_array_type(a))
    
     #y = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, x), random_mps(s))

    # If the matrix 
    if isdiag(a, 1e-10) # make back to diagonal and sqrt it
        sq_a = diag_itensor(sqrt.(array(diag(a))), is)
    else # Schur decomp is not implemented on GPU
        a = tocpu(a)
        sq_a = ITensor(sqrt(matrix(a)), is)
        sq_a = adapt(data_type, sq_a)
    end

end
