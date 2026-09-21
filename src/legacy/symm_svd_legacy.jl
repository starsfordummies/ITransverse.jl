# Superseded implementations of the complex-symmetric decompositions, kept for comparison and
# regression testing against the current ones in `ITenUtils/svd_sym.jl` and
# `ITenUtils/eig_sym.jl`.
#
# --- symm_svd ---
#
# It builds the unitary fix-up `z = U† V̄` as a full matrix and takes its *principal matrix*
# square root with `sqrt` (Schur / Björck-Hammarling) whenever `z` is not detected as
# diagonal. Two problems, both measured in `test/test_takagi_svd.jl`:
#
#  1. `sqrt(z)` is unusable here. `z` is unitary, so its eigenvalues sit *on* the unit
#     circle; a conjugate pair either side of the branch cut at -1 makes the Sylvester
#     solves singular and the result is silently garbage (`U` stays unitary, only the
#     reconstruction is wrong). This is not a corner case: a real symmetric indefinite `M`
#     gives `z` eigenvalues of exactly -1 and fails most of the time.
#  2. It is slow. The Schur square root costs ~4.5x the SVD it is correcting (1.2 s vs
#     0.25 s at n=800), and `z` itself costs a full m*n² product that the diagonal case
#     does not need at all.
#
# `symm_svd` now reads the phases off a real symmetric eigendecomposition instead
# ([`sym_unitary_sqrt`](@ref)) and only ever forms the blocks of `z` that belong to
# degenerate singular values ([`takagi_phases`](@ref)).

""" Legacy [`symm_svd`](@ref) for matrices - see the note at the top of this file. """
function symm_svd_legacy(M::Matrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M) #inclues check

    # `cutoff_on=:values` matches the current `symm_svd`: what is legacy here is the *fix-up*,
    # and sharing the truncation is what lets the equivalence tests compare just that.
    F, spec = truncated_svd(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff,
                            cutoff_on=:values)
    u,s,v = F

    z = transpose(conj(u)) * transpose(v')

    sq_z = if isapproxdiag(z)
        # If z is diagonal, just invert its diag. (`complex` added here: the original threw
        # a DomainError on a real `z`, which is exactly what a real symmetric `M` produces.)
        Diagonal(sqrt.(complex.(diag(z))))
    else
        sq_z = sqrt(z)
    end

    #sq_z should be symmetric
    uz = u * sq_z

    return SVD(uz, s, transpose(uz)), spec
end


""" Legacy [`symm_svd`](@ref) for ITensors - see the note at the top of this file. """
function symm_svd_legacy(a::ITensor, linds, rinds = uniqueinds(a, linds) ;
                         cutoff=nothing, maxdim=nothing, mindim=nothing,
                         use_absolute_cutoff=nothing, use_relative_cutoff=nothing, kwargs...)

    cL = combiner(linds)
    cR = combiner(rinds)

    ac = a * cL * cR

    iL = combinedind(cL)
    iR = combinedind(cR)

    ac = symmetrize(ac)

    # u * s * vd ≈ a  (linear cutoff, as in the current `symm_svd` - see above)
    F, spec = svd_trunc_values(ac, iL; cutoff, maxdim, mindim,
                               use_absolute_cutoff, use_relative_cutoff, kwargs...)

    z = transpose_arrows(conj(F.U) * replaceind(F.V, iR => iL))

    sq_z = sqrt(z) # block-diagonal for QN tensors

    uS = F.U * sq_z
    u = replaceinds(uS, F.v => F.u)* dag(cL)
    uS = transpose_arrows(replaceinds(uS, iL => iR)) * dag(cR)

    return ITensors.TruncSVD(u,F.S,uS, spec, F.u, F.v)
end

symm_svd_legacy(a::ITensor; kwargs...) = (@assert ndims(a) == 2; symm_svd_legacy(a, ind(a,1); kwargs...))


# --- symm_oeig ---
#
# The complex-*orthogonal* eigendecomposition. Two differences from the current version:
# its diagonality test on the Gram matrix `Z` was an absolute threshold (`norm(Z - diagz) <
# 1e-10`) rather than one relative to `norm(Z)`, and nothing reported how ill-conditioned the
# resulting basis was - `O` is complex-orthogonal, so `cond(O) = σmax(O)^2` diverges as `M`
# approaches a defective matrix, and the decomposition silently loses digits (relative error
# ~1e-6 at `σmax(O) ≈ 1.5e3`) with no indication.

""" Legacy [`symm_oeig`](@ref) - see the note above. """
function symm_oeig_legacy(M::AbstractMatrix; maxdim=nothing, cutoff=nothing,
                          use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M)
    F, spec = mytrunc_eig(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    vals = F.values
    vecs = F.vectors

    Z = transpose(vecs) * vecs

    # TODO this is a hack to enforce sqrt of diagonal matrix even when it's only approx diagonal
    diagz = Diagonal(diag(Z))

    if norm(Z - diagz) < 1e-10
        isq_z = diagz^-0.5
    else
        # isq_z_1 = Z^(-0.5)  # Old version giving GPU headaches, do it manually instead:
        Fz = eigen(Z)
        isq_z = (Fz.vectors * Diagonal(Fz.values .^ -0.5)) / Fz.vectors
    end
    O = vecs*isq_z

    return Eigen(vals, O), spec
end
