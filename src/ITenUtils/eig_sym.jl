""" Eigenvalue of matrix M with truncation. Returns Eigen() struct and spectrum
The cutoff is applied to the sum of the abs() of the eigenvalues, so that
the norm error on the truncated object is ~ sqrt(cutoff) """
function mytrunc_eig(
    M::AbstractMatrix;
    maxdim=nothing,
    mindim=1,
    cutoff=nothing,
    use_absolute_cutoff=nothing,
    use_relative_cutoff=true,
)

if any(!isfinite, M)
  throw(ArgumentError("mytrunc_eig: input matrix contains NaNs or Infs"))
end

# LAPACK geev is noticeably less reliable in single precision
if eltype(M) <: Union{Float32, ComplexF32}
  M = ComplexF64.(M)
end

DM, VM = try
  eigen(M)
catch err
  err isa LinearAlgebra.LAPACKException || rethrow()
  # rare geev non-convergence: retry once on a perturbed copy
  # (perturbation ~ eps*|M|, well below any truncation scale)
  Mp = M .+ (eps(real(float(eltype(M)))) * norm(M)) .* randn(ComplexF64, size(M))
  eigen(Mp)
end

# Sort by largest to smallest eigenvalues
p = sortperm(DM; by=abs, rev = true)
DM = DM[p]
VM = VM[:,p]

if any(!isnothing, (maxdim, cutoff))
  # truncerr from ctruncate! can be complex in corner cases, hence the abs() below
  truncerr, _ = ctruncate!(
    DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
  )
  dD = length(DM)
  if dD < size(VM, 2)
    VM = VM[:, 1:dD]
  end
else
  dD = length(DM)
  truncerr = 0.0
end

truncerr_r = abs(truncerr)
spec = Spectrum(abs.(DM), isfinite(truncerr_r) ? truncerr_r : 0.0)

return Eigen(DM, VM), spec

end

"""
    symm_oeig(M::AbstractMatrix; kwargs...) -> Eigen(vals, O), spec

Complex-*orthogonal* eigendecomposition of a complex symmetric `M`: `M = O * Λ * transpose(O)`
with `transpose(O) * O = I` 

`Z = transpose(V) * V` and `O = V * Z^(-1/2)`

 Eigenvectors belonging to distinct
eigenvalues are already complex-orthogonal - `(λi - λj) * viᵀvj = 0` - so `Z` is diagonal
unless eigenvalues coincide

`O` is complex-orthogonal rather than unitary, so it can be arbitrarily ill-conditioned:
 `cond_warn` bounds the tolerated `cond(O)`, its default chosen so that the warning fires at around a 1e-10 relative error in
`O * Λ * transpose(O)`.
"""
function symm_oeig(M::AbstractMatrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing,
                   use_relative_cutoff=nothing, cond_warn=1e3)

    M = symmetrize(M)
    F, spec = mytrunc_eig(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    vals = F.values
    vecs = F.vectors

    Z = transpose(vecs) * vecs

    isq_z = if isapproxdiag(Z; tol=1e-10)
        # `isapproxdiag` is relative to ||Z||; the previous `norm(Z - diagz) < 1e-10` was an
        # absolute threshold, which only happened to work because `eigen` hands back columns
        # of unit Euclidean norm and so bounds ||Z|| by the matrix size.
        Diagonal(inv.(sqrt.(diag(Z))))
    else
        # Z^(-1/2) as a matrix function, hence commuting with everything Z commutes with -
        # in particular Λ, which is what makes `O Λ Oᵀ = M` come out right. Done through
        # `eigen` rather than `Z^-0.5` because the latter has no GPU path.
        Fz = eigen(Z)
        (Fz.vectors * Diagonal(inv.(sqrt.(complex.(Fz.values))))) / Fz.vectors
    end

    O = vecs * isq_z

    # `transpose(O)*O = I` makes `inv(O) = transpose(O)`, so cond(O) = σmax(O)^2, bounded by
    # ||O||_F^2. Costs a reduction over O and replaces the reconstruction check that used to
    # sit here commented out (that one needed a full n³ product).
    amp = norm(O)^2 / size(O, 2)
    if !isfinite(amp) || amp > cond_warn
        @warn "symm_oeig: ill-conditioned complex-orthogonal eigenbasis (cond(O) ≳ $(round(amp; sigdigits=3))). \
               M is close to defective: as it approaches an exceptional point this decomposition \
               ceases to exist, and `O * Λ * transpose(O)` loses accuracy accordingly."
    end

    return Eigen(vals, O), spec
end


""" When called on ITensors, `symm_oeig`` returns a single `TruncEigen` object""" 
function symm_oeig(a::ITensor, linds, rinds = uniqueinds(a, linds) ; cutoff=nothing, maxdim=nothing, lefttags="eig_sym")
    no_qns_supported("symm_oeig (complex-symmetric eigendecomposition)", a)

    cL = combiner(linds)
    cR = combiner(rinds)
    am = matrix((a * cL) * cR)

    F, spec = symm_oeig(am; cutoff, maxdim)
    D = F.values
    Om = F.vectors

    eigind = Index(size(F.values,1), lefttags)
    D = diag_itensor(D, eigind, eigind')
    O = ITensor(Om, combinedind(cL), eigind) * dag(cL)
    Ot = ITensor(permutedims(Om,(2,1)), eigind', combinedind(cR)) * dag(cR)

    return ITensors.TruncEigen(D, O, Ot, spec, eigind, eigind')
end