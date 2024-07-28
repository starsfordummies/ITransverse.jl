using .ITenUtils: check_diag_matrix

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
function isid(a::ITensor, cutoff::Float64=1e-8)
    @assert ndims(a) == 2
    check_id_matrix(matrix(a), cutoff)
end


function isdiag(a::ITensor, cutoff::Float64=1e-8)
    @assert ndims(a) == 2
    check_diag_matrix(matrix(a), cutoff)
end


""" Returns pseudo-inverse of 2-dimensional ITensor (ie. matrix)
For example (for non-rectangular matrices)
A = -▷-   (largest dimension to the left)
pinv(A) =  -◁-
one should have (TODO check: contracted on the fat dimension?)
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

"""checks whether an MPO tensor is symmetric """
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


""" Easier to follow check for whether a given ITensor is symmetric in the `inds_to_permute` index pair 
Does a few allocations so it's maybe more expensive but it's meant for small tensors anyway """
function check_symmetry_itensor(T::ITensor, inds_to_permute)
  
    @assert length(inds_to_permute) == 2  # I only know how to swap pairs 

    other_inds = uniqueinds(T, inds_to_permute)

    Tten = Array(T, inds_to_permute, other_inds)
    Tten_swapped = Array(T, reverse(inds_to_permute), other_inds)


    ddelta = norm(Tten - Tten_swapped)
    if  ddelta < 1e-12
        @info("Tensor Symmetric in $inds_to_permute")
    else
        @warn("Tensor *not* symmetric normdiff=$ddelta in $inds_to_permute")
    end

end



""" build a random dxd unitary matrix as the U of an SVD of a random matrix"""
function random_unitary_svd(linds::Tuple, rind::Index)
    m = random_itensor(ComplexF64, linds..., rind)
    u, _, _ = svd(m, linds)
    return u 
end

""" TOOD WIP"""
function nonzero_elements(A::ITensor)
    return 0
end


""" hacky way to extract physical indices from an ITensor, hoping that we've been hintful enough.
- if the ITensor has only 1 dim, that's the physical dim
- if it has more than one dim, try to match the tag "phys"
- if that doesn't work, try to match the tag "Site"
- throws error otherwise
"""
function phys_ind(A::ITensor)
    if ndims(A) == 1
        physind = ind(A,1)
    else
        if !isempty(inds(A,"phys"))
            physind = inds(A,"phys")[1]
        elseif !isempty(inds(A,"Site"))
            physind = inds(A,"Site")[1]
        else
            @error "Not sure what to do here, please label physical index in tensor"
        end
    end

    return physind
end
