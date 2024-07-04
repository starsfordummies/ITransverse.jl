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

DM, VM = eigen(M)

# Sort by largest to smallest eigenvalues
p = sortperm(DM; by=abs, rev = true)
DM = DM[p]
VM = VM[:,p]

if any(!isnothing, (maxdim, cutoff))
  #println("TRUNCATING @ $maxdim, $cutoff, last eig  = $(DM[end]) ")
  truncerr, _ = ctruncate!( # ) NDTensors.truncate!!(
    DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
  )
  dD = length(DM)
  if dD < size(VM, 2)
    VM = VM[:, 1:dD]
  end
else
  #println("**NOT**TRUNCATING @ $maxdim, $cutoff,  last eig  = $(DM[end])  ")
  dD = length(DM)
  truncerr = 0.0
end

# TODO it seems that truncate!! can return complex truncerr for corner cases
spec = 0
try
  spec = Spectrum(abs.(DM), abs(truncerr))
catch e
  @error("not good, $e, $(abs.(DM)), $truncerr")
end


# TODO this doesn't work with inv() when there's truncation and VM is not square
# we could try pinv() but is it as good ? 

#M_rec = VM * Diagonal(DM) * inv(VM)

# norm_err = norm(M_rec-M)/norm(M)
# if norm_err > 1e-6
#     @warn("EIG decomp maybe not accurate, norm error $norm_err")
# end

return Eigen(DM, VM), spec

end

function symm_oeig(M::AbstractMatrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M)
    F, spec = mytrunc_eig(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    #dump(F)
    vals = F.values
    vecs = F.vectors

    Z = transpose(vecs) * vecs

    # TODO this is a hack to enforce sqrt of diagonal matrix even when it's only approx diagonal
    diagz = Diagonal(diag(Z))

    if norm(Z - diagz) < 1e-10
        isq_z = diagz^-0.5
    else
        isq_z = Z^(-0.5)
    end
    O = vecs*isq_z

    M_rec = O * Diagonal(vals) * transpose(O)

    norm_err = norm(M_rec-M)/norm(M)

    if !isnothing(cutoff)
        if norm_err > max(sqrt(cutoff), 1e-12)
            @warn("Ortho/EIG decomp maybe not accurate, norm error $norm_err (cutoff = $cutoff) sqrt=$(sqrt(cutoff))")
        else
            @debug("Ortho/EIG decomp with norm error $(norm_err) < $(sqrt(cutoff)), [norm = $(norm(M))| normS = $(norm(vals))]")
        end
    else
        @warn "No cutoff given"
    end


    return Eigen(vals, O), spec, norm_err
end





""" When called on ITensors, `symm_oeig`` returns a single `TruncEigen` object""" 
function symm_oeig(a::ITensor, linds; cutoff=nothing, maxdim=nothing)
    rinds = uniqueinds(a, linds)

    cL = combiner(linds)
    cR = combiner(rinds)
    am = matrix(a * cL * cR)

    F, spec, norm_err = symm_oeig(am; cutoff, maxdim)
    D = F.values
    Om = F.vectors

    eigind = Index(size(F.values,1), tags="eig_sym")
    D = diag_itensor(D, eigind, eigind')
    O = ITensor(Om, combinedind(cL), eigind) * dag(cL)
    Ot = ITensor(permutedims(Om,(2,1)), eigind', combinedind(cR)) * dag(cR)

    return ITensors.TruncEigen(D, O, Ot, spec, eigind, eigind')
end