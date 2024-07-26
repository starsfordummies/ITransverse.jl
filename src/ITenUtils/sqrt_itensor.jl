

""" Sqrt for a square ITensor"""
function Base.sqrt(a::ITensor, is::Tuple{<:Index, <:Index}=inds(a))

    # we want to sqrt square matrices
    # We don't want to do any funny reshaping here 
    @assert ndims(a) == 2 
    @assert dim(a,1) == dim(a,2) == dim(is[1]) == dim(is[2])

    
    # If the matrix is approx diagonal,
    if isdiag(a, 1e-10) # make back to diagonal and sqrt it
        sq_a = diag_itensor(sqrt.(array(diag(a))), is)
    else # Schur decomp for sqrt is not implemented on GPU so we need to do some back-forth..
        dmtype = promote_type(NDTensors.unwrap_array_type(a))
        a = tocpu(a)
        sq_a = adapt(dmtype, ITensor(sqrt(matrix(a)), is))
    end

    return sq_a

end
