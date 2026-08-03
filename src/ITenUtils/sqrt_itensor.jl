

""" Sqrt for a square ITensor. If it's (almost) diagonal, just returns the sqrt of its diagonal elements,
otherwise it extracts the tensor and sqrt-s it using fancy linear algebra from Julia """
function Base.sqrt(a::ITensor, is::Tuple{<:Index, <:Index}=inds(a))

    # we want to sqrt square matrices
    # We don't want to do any funny reshaping here
    @assert ndims(a) == 2
    @assert dim(a,1) == dim(a,2) == dim(is[1]) == dim(is[2])

    # A QN tensor is block diagonal and the principal square root is taken block by block
    # (densifying it here would silently drop the symmetry structure).
    hasqns(a) && return blockwise_sqrt(a)


    # If the matrix is approx diagonal,
    if isdiag(a) # make back to diagonal and sqrt it
        sq_a = diag_itensor(sqrt.(array(diag(a))), is)
    else # Schur decomp for sqrt is not implemented on GPU so we need to do some back-forth..
        dmtype = promote_type(NDTensors.unwrap_array_type(a))
        a = adapt(Array,a)
        sq_a = adapt(dmtype, ITensor(sqrt(matrix(a)), is))
    end

    return sq_a

end


"""
    blockwise_matfun(f, a::ITensor)

Apply a dense matrix function `f` (e.g. `sqrt`, `M -> M^-0.5`) to each nonzero block of a
block-diagonal (QN) matrix-like ITensor.

A matrix function of a block-diagonal matrix *is* the block-diagonal matrix of the function
applied to each block, so this is exactly the dense operation with the blocks kept intact -
densifying instead would silently drop the symmetry structure.
"""
function blockwise_matfun(f, a::ITensor)
    at  = ITensors.tensor(a)
    out = copy(at)
    for bl in nzblocks(at)
        bv = NDTensors.blockview(at, bl)
        size(bv, 1) == size(bv, 2) ||
            error("blockwise_matfun: block $(bl) is not square, size $(size(bv))")
        NDTensors.blockview(out, bl) .= f(Matrix(bv))
    end
    return itensor(out)
end

""" Principal square root of a block-diagonal (QN) matrix-like ITensor, block by block. """
blockwise_sqrt(a::ITensor) = blockwise_matfun(sqrt, a)

""" Inverse square root of a block-diagonal (QN) matrix-like ITensor, block by block. """
blockwise_invsqrt(a::ITensor) = blockwise_matfun(M -> M^-0.5, a)
