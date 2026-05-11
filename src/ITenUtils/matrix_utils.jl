""" Checks if a matrix is diagonal within a given cutoff.
Computes the off-diagonal Frobenius norm in a single allocation-free pass.
"""
function isapproxdiag(d::AbstractMatrix; tol::Float64=1e-8, verbose::Bool=false)
    T = real(eltype(d))
    off_sq  = zero(T)
    diag_sq = zero(T)
    for j in axes(d, 2), i in axes(d, 1)
        v = abs2(d[i, j])
        i == j ? (diag_sq += v) : (off_sq += v)
    end
    off_diag_norm = sqrt(off_sq)
    matrix_norm   = sqrt(diag_sq + off_sq)
    threshold = tol * max(matrix_norm, one(T))
    if off_diag_norm <= threshold
        return true
    else
        verbose && @warn "Matrix non-diagonal: ||off-diag||=$off_diag_norm, threshold=$threshold"
        return false
    end
end

isapproxid(m::AbstractMatrix; tol=1e-6) =  check_id_matrix(m; tol)


""" Check if a matrix is (proportional to) the identity within a given tol.
Accepts any AbstractMatrix (including GPU arrays and views).
"""
function check_id_matrix(m::AbstractMatrix; tol=1e-6)
    if size(m, 1) != size(m, 2)
        @warn "check_id_matrix: non-square matrix $(size(m))"
        return false
    end
    n = size(m, 1)
    T = real(eltype(m))
    # Compute max-norm deviation from I and Frobenius norm of m in one pass
    max_dev  = zero(T)
    frob_sq  = zero(T)
    for j in 1:n, i in 1:n
        v = m[i, j]
        frob_sq += abs2(v)
        expected = i == j ? one(eltype(m)) : zero(eltype(m))
        max_dev  = max(max_dev, abs(v - expected))
    end
    threshold = tol * max(sqrt(frob_sq), one(T))
    if max_dev <= threshold
        return true
    else
        @warn "Not identity: max element-wise deviation $max_dev (threshold $threshold)"
        factor = tr(m) / n
        if abs(factor) > eps(T) && isapprox(m, factor * I; atol=threshold)
            @info "Proportional to identity, factor ≈ $factor"
        end
        return false
    end
end



""" Symmetrizes a matrix to improve numerical stability (throws an error if it's not too symetric to begin with)
    TODO This fails for GPU matrices?!  """
function symmetrize(a::AbstractMatrix, tol::Float64=1e-6)
    @assert size(a,1) == size(a,2) "Matrix is not square"

    if !isapprox(a, transpose(a); atol=tol)
        @error("Not symmetric? norm(a-aT)=$(norm(a - transpose(a)))")
        #sleep(2)
    end
    return 0.5*(a + transpose(a))
end



""" build a random dxd unitary matrix as the U of an SVD of a random matrix"""
function random_unitary_svd(d::Int)
    m = rand(ComplexF64, d,d)
    u, _, _ = svd(m)
    return u 
end

"""
    haar_isometry(d_out::Int, d_in::Int)

Generate a Haar-random isometry of size (d_out, d_in),
i.e., a complex matrix V such that V'*V = I_{d_in}.
"""
function haar_isometry(d_out::Int, d_in::Int)
    @assert d_out ≥ d_in "Output dimension must be ≥ input dimension"

    # Step 1: complex Ginibre matrix
    G = randn(ComplexF64, d_out, d_in)

    # Step 2: thin QR decomposition
    Q, R = qr(G)  # Q: d_out × d_in, column-orthonormal

    # Step 3: correct phases of diagonal of R
    diagR = diag(R)
    ph = diagR ./ abs.(diagR)
    Q = Q * Diagonal(ph)

    return Matrix(Q)  # convert from Q type to plain Array
end


""" Given d^2 length, builds a vectorized identity (length d^2) """
function vectorized_identity(dsquared::Int)
    d = Int(sqrt(dsquared))
    #@assert d^2 == len "Input length must be a perfect square"
    return vec(Matrix{Float64}(I, d, d))
end



""" Return max(abs(A-B)), if quick=true it chops non-overlapping value """
function max_diff(A::Matrix, B::Matrix; quick::Bool=true)
    aR, aC = size(A)
    bR, bC = size(B)
    
    @assert aR == bR "Matrix sizes do not match: aR=$(aR) != $(bR)=bR"
    
    minC = min(aC, bC)
    
    # Max difference in overlapping region
    max_val = maximum(abs.(view(A, :, 1:minC) .- view(B, :, 1:minC)))
    
    if !quick         # Check non-overlapping columns (treated as zeros)
        if aC > bC
            max_val = max(max_val, maximum(abs.(view(A, :, (bC+1):aC))))
        elseif bC > aC
            max_val = max(max_val, maximum(abs.(view(B, :, (aC+1):bC))))
        end
    end

    return max_val
end

""" 
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

"""
Build a random n×n complex matrix with a prescribed singular value spectrum.
U, V are drawn from the Haar measure on U(n) via QR decomposition of a
random Gaussian matrix. The singular values are normalized so that
sum(s^2) = 1 (unit Frobenius norm), matching your convention.
"""
function matrix_from_spectrum(sv::Vector{Float64})
    n = length(sv)
    sv_norm = sv ./ sqrt(sum(sv .^ 2))  # normalize: sum(s_i^2) = 1

    # Haar-random unitary matrices via QR of Ginibre ensemble
    Q1, R1 = qr(randn(ComplexF64, n, n))
    Q2, R2 = qr(randn(ComplexF64, n, n))
    # QR is only Haar-uniform if we fix the sign/phase of R's diagonal
    U = Q1 * Diagonal(sign.(diag(R1)))
    V = Q2 * Diagonal(sign.(diag(R2)))

    return U * Diagonal(sv_norm) * V'
end

# ── Spectrum shapes ────────────────────────────────────────────────────────────

"""Power-law decay: s_i ~ i^(-alpha). Slow (alpha~0.5) to fast (alpha~3)."""
function spectrum_powerlaw(n::Int, alpha::Float64=1.0)
    return [i^(-alpha) for i in 1:n]
end

"""Exponential decay: s_i ~ exp(-beta * i)."""
function spectrum_exponential(n::Int, beta::Float64=0.1)
    return [exp(-beta * i) for i in 1:n]
end

"""
Step spectrum: `k` large singular values (drawn uniform on [lo_high, hi_high]),
then a hard drop to uniform noise on [lo_tail, hi_tail].
Models a low-rank signal buried in noise.
"""
function spectrum_step(n::Int, k::Int;
    lo_high=0.8, hi_high=1.2,
    lo_tail=1e-4, hi_tail=1e-3)
    sv = Vector{Float64}(undef, n)
    sv[1:k] = sort(lo_high .+ (hi_high - lo_high) .* rand(k), rev=true)
    sv[k+1:end] = lo_tail .+ (hi_tail - lo_tail) .* rand(n - k)
    return sv
end

"""
Plateau then exponential decay: `k` nearly-flat values, then fast decay.
Models a problem with a well-defined rank-k subspace but no hard cutoff.
"""
function spectrum_plateau_then_decay(n::Int, k::Int, beta::Float64=0.5)
    plateau = 1.0 .+ 0.05 .* randn(k)  # near-constant with small noise
    tail = [exp(-beta * i) for i in 1:(n-k)]
    return vcat(sort(abs.(plateau), rev=true), tail)
end

"""
Clustered spectrum: groups of singular values clustered around different scales.
`centers` is a vector of (value, count) pairs.
"""
function spectrum_clustered(n::Int,
    centers::Vector{Tuple{Float64,Int}};
    spread::Float64=0.02)
    sv = Float64[]
    for (val, count) in centers
        append!(sv, abs.(val .+ spread * randn(count)))
    end
    @assert length(sv) == n "Sum of cluster counts must equal n"
    return sort(sv, rev=true)
end

"""Flat (all singular values equal): worst case for truncation."""
function spectrum_flat(n::Int)
    return ones(n)
end

# ── Convenience constructors ───────────────────────────────────────────────────

make_powerlaw(n; alpha=1.5, kw...) =
    matrix_from_spectrum(spectrum_powerlaw(n, alpha); kw...)

make_exponential(n; beta=0.3, kw...) =
    matrix_from_spectrum(spectrum_exponential(n, beta); kw...)

make_step(n, k; kw...) =
    matrix_from_spectrum(spectrum_step(n, k; kw...); kw...)

make_plateau(n, k; beta=0.5, kw...) =
    matrix_from_spectrum(spectrum_plateau_then_decay(n, k, beta; kw...); kw...)

make_clustered(n, centers; kw...) =
    matrix_from_spectrum(spectrum_clustered(n, centers; kw...); kw...)

make_flat(n; kw...) =
    matrix_from_spectrum(spectrum_flat(n); kw...)