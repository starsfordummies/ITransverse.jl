
""" TODO Check
Builds a random matrix with a decaying singular value spectrum
"""
function randmat_decayspec(n::Int)
    @assert n > 1
    h1 = rand(ComplexF64,n,n)
    h1 = h1 + transpose(conj(h1))
    u = exp(im*h1)

    h2 = rand(ComplexF64,n,n)
    h2 = h2 + transpose(conj(h2))
    v = exp(im*h2)

    sv = rand(1)
    for jj = 2:n
        push!(sv, exp(-jj)*rand())
    end

    normalize!(sv)

    mat = u * Diagonal(sv) * v

    return mat
end

""" Checks if a matrix is diagonal within a given cutoff
"""

function check_diag_matrix(d::AbstractMatrix, cutoff::Float64=1e-8, verbose::Bool=false)

    isdiag = true

    #delta_diag = norm(d - Diagonal(d))/norm(d)
    delta_diag = sqrt(maximum(abs2, d - Diagonal(d))/maximum(abs2,diag(d)))

    if delta_diag > cutoff
        verbose && @warn "Matrix non diagonal: Δ=$(delta_diag) [cutoff=$(cutoff)]"
        isdiag = false
    end
    return isdiag
end


""" Check if a matrix is identity within a given cutoff 
"""
function check_id_matrix(m::Matrix, cutoff::Float64=1e-8)

    is_id_matrix = true 

    if size(m,1) == size(m,2)
        delta_diag = norm(m - I(size(m,1)))/norm(m)
        if delta_diag > cutoff
            @warn("Not identity: off by(norm) $delta_diag")
            if norm(m./m[1,1] - I(size(m,1)))/norm(m) < cutoff
                @info("But proportional to identity, factor $(m[1,1])")
            end
            is_id_matrix = false
        end
    else
        @error ("Not even square? $(size(m))")
        is_id_matrix = false
    end

    return is_id_matrix
end


""" Symmetrizes a matrix to improve numerical stability (throws an error if it's not too symetric to begin with)
    TODO This fails for GPU matrices?!  """
function symmetrize(a::AbstractMatrix, tol::Float64=1e-6)
    if size(a,1) != size(a,2)
        @error("Not square")
    end
    if norm(a - transpose(a))/norm(a) > tol
        @error("Not symmetric? norm(a-aT)=$(norm(a - transpose(a)))")
        #sleep(2)
    end
    return 0.5*(a + transpose(a))
end


# """ build a random dxd unitary matrix as exp(iHermitian) """
# function myrandom_unitary(d::Int)
#     m = rand(ComplexF64, d,d)
#     m = m + m'
#     u = exp(im*m)
#     return u 
# end

""" build a random dxd unitary matrix as the U of an SVD of a random matrix"""
function random_unitary_svd(d::Int)
    m = rand(ComplexF64, d,d)
    u, _, _ = svd(m)
    return u 
end



""" If we're SVD-ing a rank-2 ITensor, we can do it allocating much less memory and without needing to specify indices"""
function matrix_svd(
    A::ITensor;
    lefttags=nothing,
    righttags=nothing,
    mindim=nothing,
    maxdim=nothing,
    cutoff=nothing,
    alg=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=nothing,
    min_blockdim=nothing,
  )

  @assert ndims(A) == 2 

    lefttags = NDTensors.replace_nothing(lefttags, ts"Link,u")
    righttags = NDTensors.replace_nothing(righttags, ts"Link,v")


    USVT = svd(
        tensor(A);
        mindim,
        maxdim,
        cutoff,
        alg,
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
    )

    if isnothing(USVT)
        return nothing
    end

    UT, ST, VT, spec = USVT
    U, S, V = itensor(UT), itensor(ST), itensor(VT)

    #@show first(S), sum(S)

    #@info diag(S)
    
    u = commonind(S, U)
    v = commonind(S, V) 

    U = settags(U, lefttags, u)
    S = settags(S, lefttags, u)
    S = settags(S, righttags, v)
    V = settags(V, righttags, v)

    u = settags(u, lefttags)
    v = settags(v, righttags)


    return ITensors.TruncSVD(U, S, V, spec, u, v)
end
