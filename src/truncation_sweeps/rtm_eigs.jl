"""
Stabilised eigenvalue truncation of the reduced transition matrix (`alg = "RTMeigs"`).

`alg = "RTMeig"` keeps the dominant RTM eigenvalues but inserts the *oblique* spectral projector,
whose norm `1/cos θ` compounds over a sweep and diverges (see `rtm_eig.jl`; on the all-configs
transverse sums it blows `Σ|A|²` up to 1e12-1e32 at every χ). This variant keeps the same
*selection* -- the subspaces spanned by the dominant right and left eigenvectors -- but inserts
them as two independent isometries, exactly as the SVD route (`alg = "RTM"`) does, so every
insertion has norm 1:

    ψR side:  Qx Qx^†,  Qx = orthonormal basis of span{right eigenvectors, |λ| largest}
    ψL side:  Qy Qy^†,  Qy = orthonormal basis of span{left eigenvectors, same λ}

The price is that the kept bases of the two sides are no longer biorthonormal, so the Gram
matrix `G[β,α] = ⟨β|α⟩` (non-conjugated overlap of the ψL-side and ψR-side block states) is not
the identity and the local eigenproblem is that of the *operator*, not of the coefficient matrix:
with `ρ[α,β]` the local RTM in the enlarged bases,

    right eigenvectors (ψR coordinates):  ρ G
    left  eigenvectors (ψL coordinates):  (G ρ)ᵀ = ρᵀ Gᵀ

`G` is carried along the sweep: it is `1` at the first bond and is updated as
`G_new = Qyᵀ (G ⊗ δ_site) Qx` after each step. The eigenvalues of `ρ G` are invariant under any
invertible change of basis on the temporal sites (the RTM is only similarity-transformed), which
is the point of selecting by eigenvectors rather than singular vectors.

Diagnostic (`@debug`): the largest condition number of the carried `G`.
"""

"""
    eigs_rtm(E, R, L, Ris, Gfull; cutoff, maxdim, mindim, lefttags, righttags)
        -> (U, V, u, v, λkept)

One bond of an `RTMeigs` sweep. `ρ = E * R * L` with `Ris` the open indices of `R`;
`Gfull` the Gram matrix of the two enlarged bases, an ITensor on `Ris ∪ uniqueinds(L, E)`.
`U` (on `Ris..., u`) and `V` (on `Lop..., v`) are isometries in coordinates; use `dag(U)`,
`dag(V)` for the insertion, and `Gfull * U * V` is the next bond's Gram matrix.
"""
function eigs_rtm(E::ITensor, R::ITensor, L::ITensor, Ris, Gfull::ITensor;
        cutoff, maxdim, mindim, lefttags, righttags, rank_floor::Real = 1e-12, kwargs...)

    no_qns_supported("eigs_rtm (stabilised RTM eigenvalue truncation)", E;
        hint = "Use alg=\"RTM\" (singular values) instead.")

    Rop = Ris isa Index ? (Ris,) : Tuple(Ris)
    Lop = Tuple(uniqueinds(L, E))

    cR, cL = combiner(Rop...), combiner(Lop...)
    iR, iL = combinedind(cR), combinedind(cL)

    A  = matrix(permute(((E * R) * L) * cR * cL, iR, iL))   # ρ[α, β]
    Gm = matrix(permute(Gfull * cR * cL, iL, iR))           # G[β, α]

    FR, _ = mytrunc_eig(A * Gm)
    FL, _ = mytrunc_eig(transpose(A) * transpose(Gm))
    λ, X = FR.values, FR.vectors
    μ, Y = FL.values, FL.vectors

    m = _keep_eig(λ; cutoff, maxdim, mindim, rank_floor)
    perm = _pair_by_eigenvalue(λ, μ, m)

    Qx = _orth_basis(X[:, 1:m])
    Qy = _orth_basis(Y[:, perm])

    u = Index(size(Qx, 2), lefttags)
    v = Index(size(Qy, 2), righttags)

    U = ITensor(Qx, iR, u) * dag(cR)
    V = ITensor(Qy, iL, v) * dag(cL)

    return U, V, u, v, λ[1:m]
end

""" Orthonormal basis (in coordinates) of the column span of `X`, dropping numerically
dependent directions (near-parallel eigenvectors of a defective or clustered spectrum). """
function _orth_basis(X::AbstractMatrix; rtol::Real = 1e-12)
    F = svd(X)
    k = something(findlast(>(rtol * F.S[1]), F.S), 1)
    return F.U[:, 1:k]
end

""" `Gprev ⊗ δ(s, s''')`: Gram matrix of the enlarged bases at the next bond. `s` is the
ψR-side (plev 1) site index; its ψL-side partner carries two more primes (see `ALp`). """
_eigs_gram(Gprev, s::Index) = Gprev * delta(s, prime(s, 2))


function tlrcontract(::Algorithm"RTMeigs",
        ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        compute_ov_before::Bool = true,
        direction = :right,
        kwargs...,
    )
    L, R, sv, ovb = if direction == :right
        _tlrcontract_rtmeigs_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        _tlrcontract_rtmeigs_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
    return TruncLR(L, R, sv, compute_ov_before ? ovb : NaN, NaN)
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT. Mirrors `_tlrcontract_rtmeig_right`. """
function _tlrcontract_rtmeigs_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    AL  = AL'
    ψL  = ψL''
    ALp = replaceprime(AL, 1 => 3, tags = "Site")

    NL, NR = length(ALp), length(AR)
    n, N = min(NL, NR), max(NL, NR)
    @assert NL >= length(ψL)
    @assert NR >= length(ψR)

    sR = firstsiteinds(AR, plev = 1)
    requested_maxdim = maxdim
    ψR_out, ψL_out = typeof(ψR)(n), typeof(ψL)(n)
    S_all = zeros(ComplexF64, n - 1, requested_maxdim)
    maxcondG = 1.0

    E = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in reverse(1:N)
        env = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "$j - $(inds(env))"
    end
    ov_before = scalar(env)

    R = ψR[1] * AR[1]
    L = ψL[1] * ALp[1]
    renorm_idx = nothing
    G = ITensor(1.0)

    for j in 2:n
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        tsR = preserve_mps_tags ? (l = linkind(ψR, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], renorm_idx)
        Gfull = _eigs_gram(G, sR[j-1])

        U, V, renorm_idx, _, λv = eigs_rtm(E[j], R, L, Ris, Gfull;
            cutoff, maxdim, mindim, lefttags = tsR, righttags = tsL, kwargs...)

        ψR_out[j-1] = U
        ψL_out[j-1] = V

        G = Gfull * U * V
        maxcondG = max(maxcondG, cond(matrix(G)))

        R = dag(U) * R * get(ψR, j) * get(AR, j)
        L = dag(V) * L * get(ψL, j) * get(ALp, j)

        S_all[j-1, 1:length(λv)] .= λv ./ sum(λv)
    end

    redge_L = ITensors.OneITensor()
    for j in reverse(n+1:NL)
        redge_L *= AL[j] * get(ψL, j)
    end
    redge_R = ITensors.OneITensor()
    for j in reverse(n+1:NR)
        redge_R *= AR[j] * get(ψR, j)
    end

    ψR_out[n] = R * redge_R
    ψL_out[n] = L * redge_L

    @debug "RTMeigs right sweep: max cond(G) = $maxcondG"
    return ψL_out, ψR_out, S_all, ov_before
end


""" Builds LEFT environments, sweeps RIGHT→LEFT. Mirrors `_tlrcontract_rtmeig_left`. """
function _tlrcontract_rtmeigs_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    NL, NR = length(AL), length(AR)
    n, N = min(NL, NR), max(NL, NR)
    @assert NL >= length(ψL)
    @assert NR >= length(ψR)

    AL  = AL'
    ψL  = ψL''
    ALp = replaceprime(AL, 1 => 3, tags = "Site")

    sR = firstsiteinds(AR, plev = 1)
    requested_maxdim = maxdim
    ψR_out, ψL_out = typeof(ψR)(n), typeof(ψL)(n)
    S_all = zeros(ComplexF64, n - 1, requested_maxdim)
    maxcondG = 1.0

    E = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "Bad env[$j] ? - $(inds(env))"
    end
    ov_before = scalar(env)

    R = get(ψR, NR) * AR[NR]
    for j in reverse(n:NR-1)
        R = R * get(ψR, j) * AR[j]
    end
    L = get(ψL, NL) * ALp[NL]
    for j in reverse(n:NL-1)
        L = L * get(ψL, j) * ALp[j]
    end

    renorm_idx = nothing
    G = ITensor(1.0)

    for j in reverse(1:n-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        tsR = preserve_mps_tags ? (l = linkind(ψR, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], renorm_idx)
        Gfull = _eigs_gram(G, sR[j+1])

        U, V, renorm_idx, _, λv = eigs_rtm(E[j], R, L, Ris, Gfull;
            cutoff, maxdim, mindim, lefttags = tsR, righttags = tsL, kwargs...)

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        G = Gfull * U * V
        maxcondG = max(maxcondG, cond(matrix(G)))

        R = dag(U) * R * get(ψR, j) * AR[j]
        L = dag(V) * L * get(ψL, j) * ALp[j]

        S_all[j, 1:length(λv)] .= λv ./ sum(λv)
    end

    ψR_out[1] = R
    ψL_out[1] = L

    @debug "RTMeigs left sweep: max cond(G) = $maxcondG"
    return ψL_out, ψR_out, S_all, ov_before
end
