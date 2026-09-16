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

function symm_oeig(M::AbstractMatrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

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
        #@warn "not diag"
        # isq_z_1 = Z^(-0.5)  # Old version giving GPU headaches, do it manually instead:
        Fz = eigen(Z)
        isq_z = (Fz.vectors * Diagonal(Fz.values .^ -0.5)) / Fz.vectors 
    end
    O = vecs*isq_z

    # M_rec = O * Diagonal(vals) * transpose(O)
    #norm_err = norm(M_rec-M)/norm(M)

    # if !isnothing(cutoff)
    #     if norm_err > max(sqrt(cutoff), 1e-12)
    #         @warn("Ortho/EIG decomp maybe not accurate, norm error $norm_err (cutoff = $cutoff) sqrt=$(sqrt(cutoff))")
    #     else
    #         @debug("Ortho/EIG decomp with norm error $(norm_err) < $(sqrt(cutoff)), [norm = $(norm(M))| normS = $(norm(vals))]")
    #     end
    # else
    #     @warn "No cutoff given"
    # end


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