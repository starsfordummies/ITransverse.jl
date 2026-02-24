import LinearAlgebra: isdiag

""" Builds a random symmetric ITensor of size `(n,n)`
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


function symmetrize(a::ITensor; tol=1e-6)

    if ndims(a) != 2
        @error("Not a matrix")
    end
    if size(a,1) != size(a,2)
        @error "Not square matrix! Size is $(size(a))"
    end

    return (a + swapinds(a, inds(a)...))/2
end

""" Computes norm difference of tensor vs itself with two indices (i,j) swapped"""
function normdiff_under_swap(T::ITensor, i::Index, j::Index)
    iinds = inds(T)
    i1, i2 = findfirst(==(i), iinds), findfirst(==(j), iinds)
    perm = collect(1:length(iinds))
    perm[i1], perm[i2] = perm[i2], perm[i1]
    return norm(permute(T, iinds).tensor - permute(T, iinds[perm]).tensor) 
end

""" Checks if ITensor T is symmetric under swap of indices (i,j) (up to atol) """
function check_symmetry_swap(T::ITensor, i::Index, j::Index; atol=1e-12, verbose::Bool=true)
    norm_difference = normdiff_under_swap(T, i, j)
    if norm_difference < atol
        if verbose
            @info("Tensor Symmetric $i <-> $j")
        end
        return true 
    else
        if verbose
            @warn "Tensor *Not* Symmetric $i <-> $j, normdiff = $(norm_difference)"
        end
        return false
    end
end

"""checks whether an MPO tensor is symmetric - if we don't specify indices, try to guess from labels """
function check_symmetry_itensor_mpo(T::ITensor)
    (space_p1, space_p) = inds(T, "Site")
    (wL, wR) = inds(T, "Link")
    check_symmetry_itensor_mpo(T, wL, wR, space_p1, space_p)
end

"""checks whether an MPO tensor is symmetric, specifying the indices we want to check on"""
function check_symmetry_itensor_mpo(T::ITensor, wL::Index, wR::Index, space_p1::Index, space_p::Index)

    # check symmetry: p<->p' , wL <-> wR 
    ddelta = norm(permute(T, (wL, space_p1, space_p, wR)).tensor - permute(T, (wL, space_p, space_p1, wR)).tensor)
    if  ddelta < 1e-12
        #alternatively:
        #permute(Wc, (wL, space_p', space_p, wR)).tensor == permute(Wc, (wL, space_p, space_p', wR)).tensor
        @info("Tensor Symmetric p <->p*")
    else
        @warn("Tensor *not* symmetric p<->p*,  normdiff=$ddelta")
    end

    ddelta = norm(permute(T, (wR, space_p, space_p1, wL)).tensor - permute(T, (wL, space_p, space_p1, wR)).tensor)
    if  ddelta < 1e-12
        #permute(Wc, (wR, space_p, space_p', wL)).tensor == permute(Wc, (wL, space_p, space_p', wR)).tensor
        @info("Tensor Symmetric wL <->wR")
    else
        @warn("Tensor *not* symmetric wL<->wR, normdiff=$ddelta")
    end

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
    #@assert d^2 == len "Input length must be a perfect square"
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
to_itensor(x::ITensor, idx::Index) = replaceind(x, only(inds(x)), idx)
function to_itensor(x::ITensor, name::String)
    if length(inds(x)) == 1 
        return settags(x, name) 
    else
        return replacetags(x, "Site" => name) 
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
    m = random_itensor(i1, sim(i1))
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


""" Contracts two ITensors over combined indices iA, iB: reopens them, 
partial transposes the legs of one and contracts the two  """
function ptranspose_contract(A::ITensor, B::ITensor, iA::Index, iB::Index)

    @assert hasind(A, iA)
    @assert hasind(B, iB)
   
    dA = dim(iA)
    sqdA = isqrt(dA)

    @assert hasind(A, iA)

    temp_i1 = Index(sqdA)
    temp_i2 = Index(sqdA)


    c1 = combiner(temp_i1, temp_i2)
    c2 = combiner(temp_i2, temp_i1)

    combA = A * replaceind(c1, combinedind(c1), iA)
    combB = B * replaceind(c2, combinedind(c2), iB)

    #@show commoninds(combA, combB)

    AB = combA * combB

    return AB

end

function ITensors.sim(a::ITensor)
    replaceinds(a, inds(a) => sim.(inds(a)))
end