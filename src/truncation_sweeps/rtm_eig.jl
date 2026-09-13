"""
Eigenvalue truncation of the reduced transition matrix (`alg = "RTMeig"`), the non-symmetric
counterpart of the complex-orthogonal truncation in `truncate_sweep_sym` / `gen_canonical`.

`Algorithm"RTM"` keeps the dominant *singular* subspaces of the local RTM and inserts two
independent isometries. This keeps the dominant *eigenvalues* instead, and inserts the RTM's
oblique spectral projector

    P = Σ_{i≤m} |v_i⟩⟨w_i| ,     ⟨w_i|v_j⟩ = δ_ij

split across the two sides: the kept right eigenvectors `X` go into ψR, the kept left
eigenvectors `Y` into ψL, biorthonormalised so `YᵀX = 1`. `dag(U)`/`dag(V)` of the SVD route
are replaced by those duals, which is the only structural change to the sweep.

Why the local problem is a plain (not generalized) eigenproblem
--------------------------------------------------------------
`ρ[α,β] = Σ_{ij} R[α,i] E[i,j] L[β,j]` is the RTM written in the enlarged left-block bases of
the two sides, `α` on ψR's and `β` on ψL's. Reading off eigenvectors needs the Gram matrix
`G[α,β] = ⟨β|α⟩` of those two bases, and here it is the identity at every step:

  * at the first bond the bases are the bare (site) spaces of the two tMPOs, so `G = δ`;
  * inserting biorthogonal duals makes the kept-bond Gram exactly `1_m`, and the enlarged
    basis is (kept bond) ⊗ (site), so `G = 1_m ⊗ δ` again.

So the truncation maintains its own biorthogonal canonical form -- no separate
canonicalisation pass, and `eigen(ρ)` is all that is needed. `ρ` is square for the same
reason (both sides carry the same site dimension and the same kept bond).

!!! warning "Do not use this for an iterated contraction"
    Measured on a 100-step folded-Ising light cone for `<Z>` (`.scratch/rtm-eigenvector-truncation/`):
    this diverges at *every* bond dimension from 4 to 48, reaching O(1) error in `<Z>` by
    `t ≈ 5` where `alg = "RTM"` converges monotonically to `1.8e-5` at χ = 48. The cause is
    structural, not a tuning problem -- see below. Use `alg = "RTM"` for power methods, light
    cones, and any other repeated contraction. What this algorithm is good for is a *single*
    RTM truncation, and for studying the RTM spectrum itself.

What it buys
------------
The overlap of the truncated pair is *exactly* `Σ_{i≤m} λ_i`, so the relative error is
`|Σ_{i>m} λ_i| / |Tr τ|` -- known before truncating, where the SVD route only has a bound.
In a single truncation it matches the SVD route's accuracy and is much less sensitive to
sweep `direction`.

Why it fails under iteration
----------------------------
`P` is an *oblique* projector, so `‖P‖ = 1/cos θ > 1`, with `θ` the angle between the kept
left and right eigenspaces. An SVD truncation inserts isometries and has `‖P‖ = 1` exactly, so
it can never amplify error; here every truncation can, and over `n` contraction steps the
amplification compounds as `‖P‖^n`. On the folded Ising cone `‖P‖` was measured at 26-348
(median 84, geometric mean 87) over 48 sweeps -- i.e. ~1e93 amplification, and the T=10 run
has 200 sweeps.

The trap is that `‖P‖` is large there *despite* an essentially perfect spectrum: that cone's
RTM is numerically rank 1 (`|λ₂|/|λ₁| ~ 1e-5 … 1e-9`) with cancellation
`Σ|λ| / |Σλ| = 1.000`. Eigenvalue *separation* and eigenvector *angle* are independent, and
transverse-contraction RTMs are products of two nearly orthogonal environments, so the
spectral projector is badly conditioned even when the spectrum could not look better. Judge
this scheme by `normP`, never by the spectrum.

Two further cautions, both enforced here:
  * the relative `cutoff` (on `Σ|λ|`, as in `mytrunc_eig`) is what should pick `m`, **not** a
    bare `maxdim`; keeping more states than the numerical rank is actively harmful, unlike in
    the SVD route where extra states are merely wasted. [`_keep_eig`](@ref) caps `m` at the
    numerical rank unconditionally.
  * the insertions are oblique, so accuracy is `cond(G) * eps`, not `eps`: an *untruncated*
    `RTMeig` sweep reproduces an overlap to ~1e-10 where `alg = "RTM"` gets 1e-15.
"""

"""
    eig_rtm(E, R, L, Ris; cutoff, maxdim, mindim, lefttags, righttags)
        -> (U, D, V, DU, DV, u, v, condX)

Eigen-counterpart of [`svd_rtm`](@ref) for one bond of an RTM sweep.

`ρ = E * R * L` is the local RTM with `Ris` the open indices of `R` (ψR's side) and
`uniqueinds(L, E)` those of `L` (ψL's side). Returns

  * `U`  -- kept right eigenvectors, indices `(Ris..., u)`; goes into `ψR_out`
  * `V`  -- kept left eigenvectors, indices `(Lop..., v)`; goes into `ψL_out`
  * `D`  -- the kept eigenvalues, a diagonal ITensor on `(u, v)`
  * `DU` -- dual of `U`, indices `(u, Ris...)`, with `DU * U = δ`; use in place of `dag(U)`
  * `DV` -- dual of `V`, indices `(v, Lop...)`, with `DV * V = δ`; use in place of `dag(V)`
  * `u`, `v` -- the new kept-bond indices on the ψR and ψL sides
  * `diag` -- `(; condG, normP)`. `condG` is the condition number of the `m x m`
    biorthogonality matrix `G = Yᵀ X`; the oblique insertions amplify roundoff by roughly this
    factor, so unlike the isometric SVD route an untruncated `RTMeig` sweep is accurate to
    `cond(G) * eps`, not `eps`. `normP = ‖P‖` is the operator norm of the inserted oblique
    projector, i.e. `1/cos θ` between the kept left and right eigenspaces. **This is the
    number that decides whether the scheme is usable in an iterated contraction**: the SVD
    route has `‖P‖ = 1` exactly and cannot amplify error, while any `‖P‖ > 1` here compounds
    as `‖P‖^nsteps` over a power method or light-cone sweep.

The left eigenvectors come from a second `geev` on `transpose(ρ)`, *not* from inverting the
eigenvector matrix. That matters: `ρ` is rank deficient whenever an internal leg is smaller
than an open group (the same rank bound `svd_rtm` exploits), so the full eigenvector matrix
includes kernel directions and `inv` of it is ill-conditioned even when only the leading few
columns are wanted. Taking the `m` leading left eigenvectors directly confines the
conditioning to the `m x m` overlap `G`, which is only ill-conditioned when the kept and
discarded eigenspaces are genuinely near-parallel -- i.e. when the truncation itself is
ill-posed. Inverting `G` as a full matrix (rather than just its diagonal) also keeps
degenerate and complex-conjugate clusters consistent instead of splitting them.

Dense only: `geev` has no block-sparse implementation, and the oblique projector would not
respect the sector structure anyway.
"""
function eig_rtm(E::ITensor, R::ITensor, L::ITensor, Ris;
        cutoff, maxdim, mindim, lefttags, righttags, rank_floor::Real = 1e-12, kwargs...)

    no_qns_supported("eig_rtm (RTM eigenvalue truncation)", E;
        hint = "Use alg=\"RTM\" (singular values) instead.")

    Rop = Ris isa Index ? (Ris,) : Tuple(Ris)
    Lop = Tuple(uniqueinds(L, E))

    cR, cL = combiner(Rop...), combiner(Lop...)
    iR, iL = combinedind(cR), combinedind(cL)

    rho = ((E * R) * L) * cR * cL
    @assert dim(iR) == dim(iL) """
        eig_rtm: the two open groups must have equal dimension - the local RTM is square in a
        biorthonormal basis pair. Got $(dim(iR)) (ψR side) vs $(dim(iL)) (ψL side)."""

    A = matrix(permute(rho, iR, iL))

    # right eigenvectors from A, left eigenvectors from Aᵀ; both sorted by decreasing |λ|
    FR, _ = mytrunc_eig(A)
    FL, _ = mytrunc_eig(transpose(A))
    λ, X = FR.values, FR.vectors
    μ, Y = FL.values, FL.vectors

    m = _keep_eig(λ; cutoff, maxdim, mindim, rank_floor)

    # the two spectra are the same set but `geev` need not return them in the same order once
    # moduli are close, so pair the kept right eigenvectors with their own left partners
    perm = _pair_by_eigenvalue(λ, μ, m)
    Xm, Ym = X[:, 1:m], Y[:, perm]

    # biorthonormalise as a block: Ỹᵀ = G⁻¹ Yᵀ with G = Yᵀ X, so clusters stay consistent
    G = transpose(Ym) * Xm
    Ymt = G \ transpose(Ym)               # m x da, satisfies Ymt * Xm = 1_m
    diag = (; condG = cond(G), normP = opnorm(Xm * Ymt))

    u = Index(m, lefttags)
    v = Index(m, righttags)

    U  = ITensor(Xm, iR, u) * dag(cR)
    DU = ITensor(Ymt, u, iR) * dag(cR)
    V  = ITensor(transpose(Ymt), iL, v) * dag(cL)
    DV = ITensor(transpose(Xm), v, iL) * dag(cL)
    D  = diag_itensor(λ[1:m], u, v)

    return U, D, V, DU, DV, u, v, diag
end

""" Indices into `μ` pairing each of the first `m` entries of `λ` with its nearest partner,
each used once. Identity in the generic case; it only bites on near-degenerate clusters. """
function _pair_by_eigenvalue(λ, μ, m)
    taken = falses(length(μ))
    perm = Vector{Int}(undef, m)
    for i in 1:m
        best, bestd = 0, Inf
        for j in eachindex(μ)
            taken[j] && continue
            d = abs(λ[i] - μ[j])
            d < bestd && ((best, bestd) = (j, d))
        end
        perm[i] = best
        taken[best] = true
    end
    return perm
end

"""
    _keep_eig(λ; cutoff, maxdim, mindim, rank_floor)

How many of the (|λ|-descending) eigenvalues to keep.

`cutoff` is relative to `Σ|λ|`, the same convention as `mytrunc_eig`/`ctruncate!`.

`rank_floor` is a *hard* cap at the numerical rank: eigenvalues below `rank_floor * |λ₁|` are
never kept, whatever `maxdim` and `mindim` ask for. This is not a nicety. `ρ` is rank deficient
whenever an internal leg is smaller than an open group, so its spectrum ends in a large cluster
of numerical zeros whose eigenvectors are arbitrary; reaching into that cluster makes the
kept and discarded eigenspaces near-parallel and the oblique projector blows up. Measured on a
100-step folded-Ising light cone, driving `m` by `maxdim` alone without this floor diverged by
30 orders of magnitude. `mindim` is honoured only up to the floor.
"""
function _keep_eig(λ; cutoff, maxdim, mindim, rank_floor::Real = 1e-12)
    n = length(λ)
    n == 0 && return 0
    λmax = maximum(abs, λ)
    # hard rank cap first: λ is sorted by decreasing |λ|
    rank = λmax == 0 ? 1 : something(findlast(x -> abs(x) > rank_floor * λmax, λ), 1)

    mx = min(rank, isnothing(maxdim) ? n : maxdim)
    mn = clamp(isnothing(mindim) ? 1 : mindim, 1, mx)
    (isnothing(cutoff) || cutoff <= 0) && return mx

    tot = sum(abs, λ)
    tot == 0 && return mn
    # drop the tail whose summed |λ| is below cutoff * Σ|λ|
    tail = 0.0
    keep = n
    for i in n:-1:1
        tail += abs(λ[i])
        tail > cutoff * tot && break
        keep = i - 1
    end
    return clamp(keep, mn, mx)
end


function tlrcontract(::Algorithm"RTMeig",
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
        _tlrcontract_rtmeig_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        _tlrcontract_rtmeig_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
    return TruncLR(L, R, sv, compute_ov_before ? ovb : NaN, NaN)
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT. Mirrors `_tlrcontract_rtm_right`. """
function _tlrcontract_rtmeig_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
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
    maxcond = 1.0; maxnormP = 1.0

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

    for j in 2:n
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        tsR = preserve_mps_tags ? (l = linkind(ψR, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], renorm_idx)

        U, D, V, DU, DV, renorm_idx, _, dg = eig_rtm(E[j], R, L, Ris;
            cutoff, maxdim, mindim, lefttags = tsR, righttags = tsL, kwargs...)
        maxcond = max(maxcond, dg.condG); maxnormP = max(maxnormP, dg.normP)

        ψR_out[j-1] = U
        ψL_out[j-1] = V

        # the duals replace dag(U) / dag(V): P = U*DU is oblique, not orthogonal
        R = DU * R * get(ψR, j) * get(AR, j)
        L = DV * L * get(ψL, j) * get(ALp, j)

        λv = spectrum_vector(D)
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

    @debug "RTMeig right sweep: max cond(G) = $maxcond, max ||P|| = $maxnormP"
    maxcond > 1e8 && @warn "RTMeig: biorthogonality matrix badly conditioned (cond = $maxcond); the kept and \
        discarded eigenspaces are near-parallel, so the oblique projector is near-singular. \
        Raise cutoff (it should pick m, not maxdim)."

    # sites 1..n-1 are the oblique insertions, so the pair is NOT canonical in the usual
    # sense; the biorthogonal form lives in the pair, not in either MPS alone.
    return ψL_out, ψR_out, S_all, ov_before
end


""" Builds LEFT environments, sweeps RIGHT→LEFT. Mirrors `_tlrcontract_rtm_left`. """
function _tlrcontract_rtmeig_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
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
    maxcond = 1.0; maxnormP = 1.0

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

    for j in reverse(1:n-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        tsR = preserve_mps_tags ? (l = linkind(ψR, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], renorm_idx)

        U, D, V, DU, DV, renorm_idx, _, dg = eig_rtm(E[j], R, L, Ris;
            cutoff, maxdim, mindim, lefttags = tsR, righttags = tsL, kwargs...)
        maxcond = max(maxcond, dg.condG); maxnormP = max(maxnormP, dg.normP)

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = DU * R * get(ψR, j) * AR[j]
        L = DV * L * get(ψL, j) * ALp[j]

        λv = spectrum_vector(D)
        S_all[j, 1:length(λv)] .= λv ./ sum(λv)
    end

    ψR_out[1] = R
    ψL_out[1] = L

    @debug "RTMeig left sweep: max cond(G) = $maxcond, max ||P|| = $maxnormP"
    maxcond > 1e8 && @warn "RTMeig: biorthogonality matrix badly conditioned (cond = $maxcond); the kept and \
        discarded eigenspaces are near-parallel, so the oblique projector is near-singular. \
        Raise cutoff (it should pick m, not maxdim)."

    return ψL_out, ψR_out, S_all, ov_before
end
