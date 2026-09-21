"""  
Recall how Julia/ITensors SVD conventions work

For Julia arrays,
U, S, V = svd(M)  # => M = U * Diagonal(S) * V'  [conj transpose!]

but to build an SVD object, we pass it V' as argument, ie 
F = svd(M) = SVD(F.U, F.S, F.Vt) = SVD(F.U, F.S, (F.V)' )

For ITensors,

U, S, V = svd(T) # 

TruncSVD has no field Vt

"""


"""
    truncated_svd(M; cutoff, maxdim, cutoff_on=:squares) -> SVD, Spectrum

SVD of matrix `M` with truncation.

`cutoff_on` selects what the cutoff is measured against: `:squares` (the default, and the
ITensors convention) thresholds the discarded weight of `s^2`, `:values` the discarded
weight of `s` itself. See [`symm_svd`](@ref) for why the two differ and when each is right.
"""
function truncated_svd(
        M::AbstractMatrix;
        maxdim=nothing,
        mindim=nothing,
        cutoff=nothing,
        use_absolute_cutoff=nothing,
        use_relative_cutoff=true,
        cutoff_on::Symbol=:squares,
    ) 

    MUSV = NDTensors.svd_catch_error(M; alg=LinearAlgebra.DivideAndConquer())
    if isnothing(MUSV)
        # If "divide_and_conquer" fails, try "qr_iteration"
        alg = "qr_iteration"
        MUSV = NDTensors.svd_catch_error(M; alg=LinearAlgebra.QRIteration())
        if isnothing(MUSV)
        # If "qr_iteration" fails, try "recursive"
        alg = "recursive"
        MUSV = NDTensors.svd_recursive(M)
        end
    end
    
    if isnothing(MUSV)
        if any(isnan, M)
            println("SVD failed, the matrix you were trying to SVD contains NaNs.")
        else
            println(NDTensors.lapack_svd_error_message(""))
        end
        return nothing
    end

    MU, MS, MV = MUSV


    # What the cutoff is measured against. `:squares` is the ITensors convention - the
    # discarded weight of s², right for a one-layer wavefunction whose singular values are
    # amplitudes. `:values` is the linear one, right when they are *already* probabilities,
    # which is the case for the two-layer reduced transition matrices `symm_svd` is fed.
    # (`collect` because `truncate!!` mutates what it is given.)
    P = if cutoff_on === :values
        float.(collect(MS))
    elseif cutoff_on === :squares
        MS .^ 2
    else
        throw(ArgumentError("cutoff_on must be :values or :squares, got $(repr(cutoff_on))"))
    end
    if any(!isnothing, (maxdim, cutoff))
        P, truncerr, _ = NDTensors.truncate!!(
        P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
        )
    else
        truncerr = 0.0
    end

    spec = Spectrum(P, truncerr)
    dS = length(P)
    if dS < length(MS)
        MU = MU[:, 1:dS]
        # Fails on some GPU backends like Metal.
        # resize!(MS, dS)
        MS = MS[1:dS]
        MV = MV[:, 1:dS]
    end

    return SVD(MU,MS,MV'), spec

end



"""
    sym_unitary_sqrt(z) -> w

Principal square root of a matrix that is both **unitary and symmetric**, taken from a real
symmetric eigendecomposition instead of a Schur factorization.

Writing `z = A + im*B`, unitarity together with symmetry give `A^2 + B^2 = I` and `A*B = B*A`
with `A`, `B` real symmetric, so one *real orthogonal* `Q` diagonalizes both:
`z = Q*E*transpose(Q)` with `E = Diagonal(exp.(im*θ))`, and `w = Q*Diagonal(exp.(im*θ/2))*transpose(Q)`.

Two reasons this replaces `sqrt(z)` in [`symm_svd`](@ref):

 * `sqrt` puts its branch cut on the negative real axis, and the eigenvalues of a unitary
   matrix sit *on* the unit circle. A conjugate pair `exp(±im*(π-δ))` straddling the cut makes
   the Sylvester solves inside the Björck-Hammarling algorithm singular, and the answer is
   silently wrong (error 1e-3 at δ=1e-12, O(1) at δ=0). Reading the phases off an
   eigendecomposition is exact whatever the branch. A real symmetric indefinite `M` produces
   eigenvalues of exactly -1 here, so this is a routine input, not a corner case.
 * one real symmetric eig instead of a complex Schur decomposition (~4x cheaper).

The result is a genuine *matrix function* of `z` - within an eigenspace the choice of `Q` is
immaterial - so it commutes with everything `z` commutes with (in particular the singular
values) and preserves any block structure `z` has, QN blocks included.
"""
function sym_unitary_sqrt(z::AbstractMatrix; cluster_tol=1e-7, diag_tol=1e-12)
    n = size(z, 1)
    n == 1 && return fill(sqrt(complex(z[1, 1]) / abs(z[1, 1])), 1, 1)

    # fast path: no degeneracy among the corresponding singular values leaves z diagonal
    if isapproxdiag(z; tol=diag_tol)
        d = diag(z)
        return Matrix(Diagonal(sqrt.(complex.(d) ./ abs.(d))))
    end

    F = eigen(Symmetric(real.(z)))
    Q, λ = F.vectors, F.values

    # `real(z)` alone does not fix Q inside a cluster of equal eigenvalues - which happens
    # whenever two phases are ±θ. `imag(z)` commutes with it, so one pass of diagonalizing
    # that inside each cluster completes the simultaneous diagonalization (and no further
    # pass is needed: cos θ determines the phase up to its sign, sin θ then fixes it).
    B = imag.(z)
    i = 1
    while i <= n
        j = i
        while j < n && λ[j + 1] - λ[i] <= cluster_tol
            j += 1
        end
        if j > i
            Qc = Q[:, i:j]
            Q[:, i:j] = Qc * eigen(Symmetric(transpose(Qc) * B * Qc)).vectors
        end
        i = j + 1
    end

    e = vec(sum(Q .* (z * Q); dims=1))          # = diag(transpose(Q) * z * Q), the phases
    return Q * Diagonal(sqrt.(complex.(e) ./ abs.(e))) * transpose(Q)
end


"""
    degenerate_blocks(s; rtol=1e-7) -> Vector{UnitRange{Int}}

Contiguous runs of the (descending) singular values `s` that are degenerate within
`rtol*s[1]`. Each run is compared against its own first entry rather than its predecessor, so
a slowly decaying tail is not chained into one huge block.

Grouping too generously only costs time, never accuracy - the fix-up `z` is block diagonal
over *exactly* these groups, and an off-diagonal element of `z` linking singular values a
distance `g` apart is of size `eps*s[1]/g`, so anything left ungrouped at `rtol=1e-7` is at
the 1e-9 level.
"""
function degenerate_blocks(s::AbstractVector; rtol=1e-7)
    n = length(s)
    tol = rtol * (isempty(s) ? one(eltype(s)) : abs(s[1]))
    blocks = UnitRange{Int}[]
    i = 1
    while i <= n
        j = i
        while j < n && abs(s[i] - s[j + 1]) <= tol
            j += 1
        end
        push!(blocks, i:j)
        i = j + 1
    end
    return blocks
end


"""
    takagi_phases(u, v, s; degen_tol=1e-7) -> w

The unitary fix-up `w` that turns an ordinary SVD `M = u*Diagonal(s)*v'` of a complex
*symmetric* `M` into its Takagi factorization: `M = (u*w) * Diagonal(s) * transpose(u*w)`.

`w` is a square root of `z = u' * conj(v)`, which is unitary, symmetric, and block diagonal
over groups of degenerate singular values. Only those blocks are ever formed or
square-rooted: with distinct singular values `z` is diagonal, `w` is a diagonal of phases
whose entries come out of a single reduction, and the whole fix-up is O(mn) - against the
m*n² product needed to build `z` plus an O(n³) Schur square root if it is done naively.

`abs(z[i,i]) == 1` exactly when column `i` decouples (`z` is unitary, so its rows have norm
1), which is what the singleton check below tests. It is a backstop, not the main criterion:
`1 - abs(z[i,i])` goes like the *square* of the off-diagonal weight it is meant to catch, so
it only sees what `degen_tol` misses by a wide margin. The grouping itself is what keeps the
dropped off-diagonal elements at the 1e-9 level.
"""
function takagi_phases(u::AbstractMatrix, v::AbstractMatrix, s::AbstractVector;
                       degen_tol=1e-7, warn_tol=1e-11)
    n = length(s)
    dz = conj.(vec(sum(u .* v; dims=1)))        # diag(u' * conj(v)), without the m*n² product
    blocks = degenerate_blocks(s; rtol=degen_tol)

    check_singleton(i) = abs(abs(dz[i]) - 1) <= warn_tol ||
        @warn "takagi_phases: |z[$i,$i]| = $(abs(dz[i])) ≠ 1, so singular value $i is mixed \
               with another one that degen_tol=$degen_tol did not group with it"

    if length(blocks) == n                       # no degeneracy at all: z is diagonal
        foreach(check_singleton, 1:n)
        return Diagonal(sqrt.(complex.(dz) ./ abs.(dz)))
    end

    w = zeros(complex(eltype(u)), n, n)
    for r in blocks
        if length(r) == 1
            i = first(r)
            check_singleton(i)
            w[i, i] = sqrt(complex(dz[i]) / abs(dz[i]))
        else
            w[r, r] = sym_unitary_sqrt(u[:, r]' * conj(v[:, r]); cluster_tol=degen_tol)
        end
    end
    return w
end


"""
Symmetric (Takagi) SVD decomposition of a matrix. Returns `SVD(Uz, S, transpose(Uz)), spec`.

    F = symm_svd(M) ; F.U * Diagonal(F.S) * transpose(F.U) ≈ M # true

**The cutoff is applied to the discarded sum of the singular values, not to the sum of their
squares.** Every caller feeds `symm_svd` a *reduced transition matrix* - two layers of the
state, as built in `truncation_sweeps/sweeps_sym.jl` - whose singular values are already
probabilities rather than amplitudes. Squaring them thresholds `p²`, which discards at
`sqrt(cutoff)`: with the ITensors convention `cutoff=1e-12` throws away everything below
`1e-6`. That is the same convention [`mytrunc_eig`](@ref)/[`symm_oeig`](@ref) and the
`densitymatrix` contraction already use, so one `cutoff` now means one thing on all three
truncation routes. Pass `cutoff_on=:squares` to [`truncated_svd`](@ref) for the plain
ITensors behaviour.
"""
function symm_svd(M::Matrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M) #inclues check

    F, spec = truncated_svd(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff,
                            cutoff_on=:values)
    u, s, v = F

    # M = u S v' with M symmetric makes z = u' * conj(v) unitary, symmetric and commuting
    # with S, so M = (u √z) S transpose(u √z). See `takagi_phases` for how √z is built, and
    # `src/legacy/symm_svd_legacy.jl` for the previous (Schur-based) version.
    uz = u * takagi_phases(u, v, s)

    # here we should have m = uz * S * transpose(uz)
    # so Vd = transpose(uz)  (?)
    # but then be careful, cause unpacking this will return uz,s,conj(uz)
    return SVD(uz, s, transpose(uz)), spec
end


"""
    svd_trunc_values(ac::ITensor, iL; cutoff, maxdim, mindim, kwargs...) -> F, spec

`svd` with the cutoff measured against the discarded **sum of the singular values** rather
than the sum of their squares - the `:values` convention of [`truncated_svd`](@ref), which
ITensors' own `svd` does not offer (it squares unconditionally, `NDTensors` dense
`linearalgebra.jl` and `blocksparse/linearalgebra.jl` alike).

Both criteria keep the *same* set, the largest singular values, and differ only in how many,
so the cut is applied by re-running `svd` with `maxdim` set to the count the linear rule
asks for. Going back through `svd` rather than slicing is what keeps this right for QN
(block-sparse) tensors, where the retained values are spread across blocks and the index has
to be rebuilt with them; the second pass is skipped whenever the cutoff does not bite, which
includes every call that is limited by `maxdim` alone.
"""
function svd_trunc_values(ac::ITensor, iL; cutoff=nothing, maxdim=nothing, mindim=nothing,
                          use_absolute_cutoff=nothing, use_relative_cutoff=nothing, kwargs...)

    # no cutoff on this pass: it is capped by maxdim only
    F = svd(ac, iL; maxdim, mindim, kwargs...)

    s = float.(collect(spectrum_vector(F.S)))

    (isnothing(cutoff) || iszero(cutoff)) && return F, Spectrum(s, 0.0)

    truncerr, _ = ctruncate!(s; mindim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    spec = Spectrum(s, abs(truncerr))

    length(s) == dim(F.u) && return F, spec
    n = length(s)

    # Without QNs the singular values come back globally sorted, so the retained set is just
    # the leading `n` columns and the cut is a slice of what has already been computed. With
    # QNs they are scattered over blocks and the new link has to carry the right sector
    # dimensions, which is what going back through `svd` builds (~1.6x the decomposition).
    if !hasqns(ac) && ndims(ac) == 2
        iR = only(uniqueinds(ac, iL))
        u2 = Index(n, tags(F.u))
        v2 = Index(n, tags(F.v))
        U2 = ITensor(matrix(F.U, iL, F.u)[:, 1:n], iL, u2)
        V2 = ITensor(matrix(F.V, iR, F.v)[:, 1:n], iR, v2)
        return ITensors.TruncSVD(U2, diag_itensor(s, u2, v2), V2, spec, u2, v2), spec
    end

    return svd(ac, iL; maxdim=n, mindim, kwargs...), spec
end


"""
    symm_svd(a::ITensor, linds, rinds = uniqueinds(a, linds); kwargs...)

Complex-*symmetric* SVD: returns `TruncSVD(U, S, Uᵀ, ...)` with `a ≈ U * S * transpose(U)`,
built from an ordinary SVD plus the unitary fix-up `z = U† V` (`a = U √z · S · (U √z)ᵀ`).

Works with QNs: the ordinary `svd` and the combiners are block-sparse already, `√z` is taken
block by block ([`sym_unitary_sqrt`](@ref) through [`blockwise_matfun`](@ref) - each QN block
of `z` is itself unitary and symmetric) and the two `transpose_arrows` reverse the arrows
that the transposition implies. Both are exact no-ops without QNs, so the plain path is
unchanged - see `test_qn_symmetric.jl` for the reconstruction checks.

The cutoff is linear in the singular values rather than in their squares, as for the matrix
method - see [`symm_svd(::Matrix)`](@ref) for why, and [`svd_trunc_values`](@ref) for how.
"""
function symm_svd(a::ITensor, linds, rinds = uniqueinds(a, linds) ;
                  cutoff=nothing, maxdim=nothing, mindim=nothing,
                  use_absolute_cutoff=nothing, use_relative_cutoff=nothing, kwargs...)

    cL = combiner(linds)
    cR = combiner(rinds)

    ac = a * cL * cR

    iL = combinedind(cL)
    iR = combinedind(cR)

    ac = symmetrize(ac)

    # u * s * vd ≈ a
    F, spec = svd_trunc_values(ac, iL; cutoff, maxdim, mindim,
                               use_absolute_cutoff, use_relative_cutoff, kwargs...)

    # z = U^dag * V as a (u,v) matrix. With QNs `dag(U)` would flip the arrow of the leg we
    # contract over, so conjugate the data instead and reverse the arrows afterwards - that
    # is exactly what the transposition in `a = U S Uᵀ` demands.
    z = transpose_arrows(conj(F.U) * replaceind(F.V, iR => iL))

    # `sqrt(z)` (Schur) is both slower and numerically unusable here - z is unitary, so its
    # eigenvalues lie on the unit circle, right on the branch cut. `sym_unitary_sqrt` is a
    # genuine matrix function, so it still commutes with S and preserves the QN blocks, and
    # applying it block by block is exactly the dense operation with the blocks kept intact.
    # (`complex` because the phases are complex even when z itself came out real, which is
    # what a real symmetric environment gives; `blockwise_matfun` writes into a copy of z.)
    sq_z = hasqns(z) ? blockwise_matfun(sym_unitary_sqrt, complex(z)) :
                       ITensor(sym_unitary_sqrt(matrix(z)), inds(z))

    uS = F.U * sq_z
    u = replaceinds(uS, F.v => F.u)* dag(cL)
    uS = transpose_arrows(replaceinds(uS, iL => iR)) * dag(cR)

    return ITensors.TruncSVD(u,F.S,uS, spec, F.u, F.v)
end


""" If we don't specify linds, assume we're working with a matrix and just do index1 vs index2 """
function symm_svd(a::ITensor; kwargs...)
    @assert ndims(a) == 2
    symm_svd(a, ind(a,1); kwargs...)
end




""" Using SVD, split a symmetric tensor in the product of two symmetric ones  """
function symm_factorization(a::ITensor, linds; cutoff=nothing, maxdim=nothing)
    rinds = uniqueinds(a, linds)

    cL = combiner(linds)
    cR = combiner(rinds)

    ac = a * cL * cR

    iL = combinedind(cL)
    iR = combinedind(cR)

    #ac = symmetrize(ac)

    # u * s * vd ≈ a 
    u,s,vd, spec = svd(ac, iL; cutoff, maxdim)
   
    index_u = commonind(u,s)
    index_v = commonind(vd,s)

    # The fix-up here is `z = u† conj(V)` exactly as in `symm_svd` - unitary, symmetric, and
    # commuting with S. The previous version folded S into it and took a Schur `sqrt` of the
    # product, which sits on its branch cut (the phases are those of a unitary matrix); since
    # z commutes with S, `sqrt(z*S)` is just `sqrt(z)` times the positive `sqrt(S)`, and the
    # unitary part has an exact square root (see [`sym_unitary_sqrt`](@ref)).
    z = dag(u) * vd' * delta(iL, iR') * delta(index_v', index_v)

    sq_z = ITensor(sym_unitary_sqrt(matrix(z, index_u, index_v)) * Diagonal(sqrt.(diag(s))),
                   index_u, index_v)

    uu = u * sq_z
    uuL = uu * dag(cL)
    uuR = uu * delta(iL, iR) * dag(cR)
  
    return uuL, uuR
end
