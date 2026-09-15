import LinearAlgebra: isdiag

""" Builds a random complex symmetric (not hermitian) ITensor of size `(n,n)`
"""
function randsymITensor(n::Int) :: ITensor
    a = rand(ComplexF64,n,n)
    as = a + transpose(a)
    i1 = Index(n,tags="left")
    i2 = Index(n,tags="right")
    at = ITensor(as, i1, i2)

    return at
end

""" TODO Check
Builds a random square ITensor with a decaying singular value spectrum
"""
function randITensor_decayspec(n::Int)
    mat = randmat_decayspec(n)
    return ITensor(mat, Index(n,"left"), Index(n,"right"))
end




""" Check if an ITensor is identity within a given cutoff 
"""
function isid(a::ITensor, tol::Float64=1e-8)
    @assert ndims(a) == 2
    check_id_matrix(matrix(a); tol)
end


function LinearAlgebra.isdiag(a::ITensor)
    @assert ndims(a) == 2
    isapproxdiag(matrix(a); tol=1e-8)
end


""" Returns pseudo-inverse of 2-dimensional ITensor (ie. matrix)
For example (for non-rectangular matrices)
A = -▷-   (largest dimension to the left)
pinv(A) =  -◁-
one should have (contracted on the fat dimension)
pinv(A) * A = I   -◁--▷-  = ----
""" 
function pinvten(a::ITensor, check::Bool=true)
    @assert ndims(a) == 2
    #@assert size(a,1) == size(a,2)  # not necessary for pinv ?!
    if isa(a.tensor, NDTensors.DiagTensor) 
        ainv = a.^(-1)
    else
        ainv = ITensor( pinv(matrix(a)), ind(a,2), ind(a,1))  # swap indices
    end

    if check
        check_id_matrix(matrix(ainv * prime(a, ind(a,2)) ))
    end

    return ainv
end



"""
    transpose_matrix(a::ITensor, i::Index, j::Index)

Transpose of a matrix-like ITensor in its `(i,j)` legs.

With QNs `i` and `j` are dual, so a plain `swapinds` would leave the arrows inconsistent and
even `a - swapinds(a,i,j)` is not a well-formed subtraction. Reversing the arrows first
([`transpose_arrows`](@ref)) and then swapping the labels back gives a tensor carrying
*exactly* the original indices with the data transposed. Reduces to `swapinds(a, i, j)`
without QNs.
"""
transpose_matrix(a::ITensor, i::Index, j::Index) =
    hasqns(a) ? swapinds(transpose_arrows(a), (i, j), (j, i)) : swapinds(a, (i,), (j,))

function symmetrize(a::ITensor; tol=1e-6, check=true)
    i, j = inds(a)

    if dim(i) != dim(j)
        error("Not a square matrix! Dimensions are $(dim(i)) × $(dim(j))")
    end

    a_T = transpose_matrix(a, i, j)

    if check
        asym = norm(a - a_T) / norm(a)
        asym > tol && @error("ITensor is not symmetric: relative asymmetry $asym > tol=$tol")
    end

    return (a + a_T) / 2
end


"""
    spectrum_vector(S::ITensor)

The diagonal of `S` (singular values or eigenvalues) as a plain vector, in *global*
descending order of magnitude.

Needed because with QNs the storage of a diagonal tensor is laid out block by block, so the
raw `storage(S).data` is a concatenation of per-block spectra and is not globally ordered:
when block dimensions shift between iterations the concatenated vector reorders, which makes
iteration-to-iteration comparisons (the power-method `ds`) spike spuriously. Without QNs the
values already come out ordered and this returns them untouched.
"""
function spectrum_vector(S::ITensor)
    v = Array(storage(S).data)
    return hasqns(S) ? sort(v; by=abs, rev=true) : v
end

""" Raise a clear error when a routine that has no block-sparse implementation is handed
QN-conserving tensors. Without this they would silently densify (returning tensors with a
mix of QN and plain indices), which is far worse than failing. """
function no_qns_supported(what::AbstractString, x; hint="Use alg=\"naive\" or \"densitymatrix\" instead, or build the chain without QNs.")
    hasqns(x) && error("""
        $(what) has no QN (block-sparse) implementation: it goes through a dense matrix
        decomposition and would silently drop the symmetry structure.
        $(hint)""")
    return x
end

"""
    transpose_arrows(A)

Reverse the QN arrows of `A` (ITensor, MPS or MPO) **without** conjugating its data, ie.
`dag ∘ conj`. This is what turns a ket into the corresponding "transposed bra" used by all
the no-conjugation contractions here (⟨ψ*|ψ⟩ and friends): with QNs a state cannot be
contracted with itself, since both copies carry the same arrows.

Exactly the identity for objects without QNs, so it can be applied unconditionally - and in
that case the input is returned *as is*, since `dag(conj(x))` would otherwise copy the data
twice to reproduce it (the callers only read the result). That matters: these helpers sit in
the inner loops of the RTM sweeps, where the copies cost ~10% of the allocations.
"""
transpose_arrows(A::ITensor) = hasqns(A) ? dag(conj(A)) : A
transpose_arrows(M::AbstractMPS) = hasqns(M) ? dag(conj(M)) : M

""" `true` if `ll` and `rr` cannot be contracted site-by-site because their site indices
carry the same QN arrows (in which case one of them needs [`transpose_arrows`](@ref)). """
function arrows_clash(ll::TMPSorMPS, rr::TMPSorMPS)
    ll, rr = unsided(ll), unsided(rr)
    (hasqns(ll) && hasqns(rr)) || return false
    return any(dir(a) == dir(b) for (a, b) in zip(allsiteinds(ll), allsiteinds(rr)))
end

"""
    arrow_match(stored::Index, target::Index)

Return `target` (or `dag(target)`) carrying the arrow that lets it stand in for `stored`,
ie. the one for which `delta(dag(stored), arrow_match(stored, target))` is a legal QN delta.
No-op for indices without QNs, so it can be sprinkled unconditionally.
"""
arrow_match(stored::Index, target::Index) =
    (hasqns(stored) && hasqns(target) && dir(stored) != dir(target)) ? dag(target) : target

""" The leg of `T` that matches `i` by id and prime level, ie. `i` *as stored* in `T`
(the same index, but possibly carrying the opposite arrow). """
function stored_ind(T::ITensor, i::Index)
    # iterate the index tuple directly: `collect`ing it here would allocate on every leg of
    # every site, which shows up in the builders
    for x in inds(T)
        (id(x) == id(i) && plev(x) == plev(i)) && return x
    end
    return error("index $(i) not found in tensor with inds $(inds(T))")
end

"""Computes norm difference of tensor vs itself with two indices (i,j) swapped"""
function normdiff_under_swap(T::ITensor, i::Index, j::Index)
    return norm(T - swapinds(T, (i,), (j,)))
end

""" Checks if ITensor T is symmetric under swap of indices (i,j) (up to atol).

Skipped for QN tensors: the two legs carry dual arrows (and generally differently ordered
QN blocks), so `T - swapinds(T,i,j)` is not even a well-formed subtraction there. Returns
`missing` in that case - it is a diagnostic only, nothing downstream depends on the value.
"""
function check_symmetry_swap(T::ITensor, i::Index, j::Index; atol=1e-12, verbose::Bool=true)
    if hasqns(T)
        verbose && @info "Symmetry check skipped for QN tensor ($i <-> $j carry dual arrows)"
        return missing
    end
    norm_diff = normdiff_under_swap(T, i, j)
    is_sym = norm_diff < atol
    if verbose
        is_sym ? @info("Tensor symmetric $i <-> $j") :
                 @warn("Tensor *not* symmetric $i <-> $j, normdiff = $norm_diff")
    end
    return is_sym
end

"""checks whether an MPO tensor is symmetric - if we don't specify indices, try to guess from labels """
function check_symmetry_itensor_mpo(T::ITensor)
    (space_p1, space_p) = inds(T, "Site")
    (wL, wR) = inds(T, "Link")
    check_symmetry_itensor_mpo(T, wL, wR, space_p1, space_p)
end

"""checks whether an MPO tensor is symmetric, specifying the indices we want to check on"""
function check_symmetry_itensor_mpo(T::ITensor, wL::Index, wR::Index, space_p1::Index, space_p::Index)
    check_symmetry_swap(T, space_p1, space_p; atol=1e-12)
    check_symmetry_swap(T, wL, wR; atol=1e-12)
end





""" build a random dxd unitary matrix as the U of an SVD of a random matrix"""
function random_unitary_svd(linds::Tuple, rind::Index)
    m = random_itensor(ComplexF64, linds..., rind)
    u, _, _ = svd(m, linds)
    return u 
end

function haar_isometry(linds, rinds)

    Ldim = prod(dim.(linds))
    Rdim = prod(dim.(rinds))

    M = haar_isometry(Ldim, Rdim)

    return ITensor(M, linds..., rinds...)

end


""" hacky way to extract physical indices from an ITensor, hoping that we've been hintful enough.
- if the ITensor has only 1 dim, that's the physical dim
- if it has more than one dim, try to match the tag "phys"
- if that doesn't work, try to match the tag "Site"
- throws error otherwise
"""
function phys_ind(A::ITensor)
    physind = if ndims(A) == 1
        ind(A,1)
    else
       only(inds(A, "Site"))
    end

    return physind
end

""" Given an index, builds an ITensor containing vectorized identity of the appropriate size """
function vectorized_identity(ind::Index)
    d = Int(sqrt(dim(ind)))
    @assert d^2 == dim(ind) "Index dimension must be a perfect square, got $(dim(ind))"
    return ITensor(vec(Matrix{Float64}(I, d, d)), ind)
end

""" just unwraps t.tensor.storage.data - No checks are made! """
function itensor_to_vector(t::ITensor)
    # Just unwrap it 
    return storage(t).data 
end

# Core conversion: Vector → ITensor
to_itensor(x::AbstractVector, idx::Index) = ITensor(complex(x), idx)
to_itensor(x::AbstractVector, name::String="v") = ITensor(complex(x), Index(length(x), name))

# ITensor retagging/reindexing
to_itensor(x::ITensor) = x
to_itensor(x::ITensor, idx::Index) = replaceind(x, only(inds(x)), idx)
function to_itensor(x::ITensor, name::String)
    if length(inds(x)) == 1 
        return settags(x, name) 
    else
        return settags(x, name, only(inds(x, "Site")))
        #replacetags(x, "Site" => name) 
    end
end

""" This is maybe not too fast but should be general and generalizable enough.
Given an operator as string, like "X" or "Sp", builds it for the input physical site and returns an Array with its (vectorized) elements.  """
function vectorized_op(operator, site)
   itensor_to_vector(ITensors.op(operator, site))
end

function random_uni(i1::Index)
    m = random_itensor(i1, sim(i1))
    q, _ = qr(m, i1)
    return q 
end

function random_iso(i1::Index, i2::Index)
    @assert dim(i1) >= dim(i2)
    m = random_itensor(i1, i2)
    u,s,v = svd(m, i1)
    return u,s,v
end

""" Extends ITensors' contract() to compute the product of two tensors along given indices
Basically a shorthand for t1 * replaceind(t2, i2 => i1) 
"""
function ITensors.contract(t1::ITensor, t2::ITensor, i1::Index, i2::Index)
    @assert hasind(t1, i1)
    @assert hasind(t2, i2)
    return t1 * replaceind(t2, i2 => i1)
end

function ITensors.sim(a::ITensor)
    replaceinds(a, inds(a), sim.(inds(a)))
end

ITensors.ndims(::ITensors.OneITensor) = 1


""" finds the dominant eigenvector of A (matrix as ITensor) in the direction given by the index `j` """ 
function dominant_eigenvectors(A::ITensor, j::Index; howmany::Int=1, which=:LM, kwargs...)
    # A must have exactly 2 indices
    @assert length(inds(A)) == 2
    @assert hasind(A, j)

    i = uniqueind(A, j)

    A = replaceinds(A, (i,j) => (i',i))
    
    x0 = random_itensor(i) 

    vals, vecs, info = eigsolve(A, x0, howmany, which; kwargs...)
    
    # @show j
    # @show inds(A)
    # @show inds(x0)
    # @show inds(vecs[1])

    vals[1:howmany], vecs[1:howmany], info
end


""" ITensors gives right eigenvectors AR = RL so R natually has rind of A,
here we return the decomposition 
` (lind)-R-(eig)-Λ-(eig')-Rdag-(rind) ≈ A `
for hermitian matrices encoded as ITensors """
function eigdecomp_mat(a::ITensor, lind; ishermitian, kwargs...)
    @assert ndims(a) == 2 
    @assert dim(a,1) == dim(a,2)
    @assert ishermitian == true "non-herm not implemented yet"
    rind = uniqueind(a,lind)
    vals, vecs = eigen(a, lind, rind, kwargs...)
    lambda_ind = commonind(vals,vecs)
    R = replaceind(vecs, rind => lind)
    Rd = prime(dag(vecs), lambda_ind)

    return R, vals, Rd 
end
