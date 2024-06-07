
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


""" SVD of matrix M with truncation. Returns SVD() object and spectrum """
function mytrunc_svd(
        M::Matrix;
        maxdim=nothing,
        mindim=nothing,
        cutoff=nothing,
        use_absolute_cutoff=nothing,
        use_relative_cutoff=true,
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


    # The truncation - first we square the SVs since they should sum^2 = 1 
    # then discard SVs so that sum(SV^2[cut:end] < cutoff)

    P = MS .^ 2
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

    rec_M = MU * Diagonal(MS) * MV'

    #@show isapprox(rec_M, M, rtol=cutoff)
    #@show norm(rec_M - M)/norm(M)
    #@show truncerr, truncerr^0.5

    # hacky but
#     if isnothing(cutoff)
#         cutoff = 1e-14
#    end
    # if LinearAlgebra.norm2(rec_M - M)/LinearAlgebra.norm2(M) > cutoff && size(MS,1) < maxdim
    #     @warn("SVD decomp maybe not accurate, norm error $(norm(rec_M - M)/norm(M)) > $(sqrt(truncerr))")
    # end

    # # ! TODO CHECK: Do we need to put MV or conj(transpose(MV) here??
    return SVD(MU,MS,MV'), spec

end

function check_mytrunc_svd(f::SVD, M::Matrix, cutoff::Float64)

    rec_M = f.U * Diagonal(f.S) * f.V'
    delta_norm2 = LinearAlgebra.norm2(rec_M - M)/LinearAlgebra.norm2(M) 
    if delta_norm2 > cutoff 
        @warn("SVD decomp maybe not accurate, norm2 error $(delta_norm2) > $(cutoff)")
        return delta_norm2
    end

    return 0
end



""" Symmetric SVD decomposition of a matrix. 
Returns SVD(Uz, S, Uz^T), spec, norm_err
Remember that the cutoff is applied to the sum of the squares of the singular values, so that 
the norm error on the truncated object is ~ sqrt(cutoff)
F = symm_svd(M) ; F.U * Diagonal(F.S) * transpose(F.U) ≈ M # true
"""
function symm_svd(M::Matrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M) #inclues check 

    F, spec = mytrunc_svd(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    u,s,v = F
    

    #@show isapprox(u*Diagonal(s)*vd, M)
    #@show isapprox(u*Diagonal(s)*conj(transpose(vd)), M)

    # z = u^* vdag 
    z = transpose(conj(u)) * transpose(v')
    #@show z   # should be at most block-diag
    #sq_z = sqrt.(Diagonal(diag(z)))
    # If z is diagonal, just invert its diag 
    diagz = Diagonal(diag(z))
    # TODO this is a hack to enforce sqrt of diagonal matrix even when it's only approx diagonal
    if norm(z - diagz) < 1e-10
        sq_z = diagz^0.5
    else
        sq_z = z^0.5
    end

    # sq_z should be symmetric 
    if norm(sq_z - transpose(sq_z))/norm(sq_z) > 1e-8
        @error "sqrt(z) not symmetric? "
        @show norm(z - transpose(z))
        @show norm(z - Diagonal(diag(z)))
        @show norm(sq_z - transpose(sq_z)), norm(sq_z - transpose(sq_z))/norm(sq_z)
        @show z 
    end

    #uz = u * Diagonal(transpose(sq_z))
    uz = u * (transpose(sq_z))

    M_rec = uz * Diagonal(s) * transpose(uz)

    # hacky but 
    if isnothing(cutoff)
         cutoff = 1e-14
    end
    if isnothing(maxdim)
        maxdim = size(M,1)
    end

    norm_err = norm(M_rec-M)/norm(M)
    if norm_err > sqrt(cutoff) && size(s,1) < maxdim
        @warn("Symmetric SVD decomp maybe not accurate, norm error $(norm_err) > $(sqrt(cutoff))")
    else
        @debug("Symmetric SVD decomp with norm error $(norm_err) < $(sqrt(cutoff))")
    end

    # here we should have m = u * S * uT
    # so Vd = uT  (?)
    # but then be careful, cause unpacking this will return u,s,conj(u) 
    return SVD(uz, s, transpose(uz)), spec, norm_err
end



""" Eigenvalue of matrix M with truncation. Returns Eigen() struct and spectrum
The cutoff is applied to the sum of the abs() of the eigenvalues, so that 
the norm error on the truncated object is ~ sqrt(cutoff) """
function mytrunc_eig(
    M::Matrix;
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

function symm_oeig(M::Matrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

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
        if norm_err > sqrt(cutoff) 
            @warn("Ortho/EIG decomp maybe not accurate, norm error $norm_err (cutoff = $cutoff) sqrt=$(sqrt(cutoff))")
        else
            @debug("Ortho/EIG decomp with norm error $(norm_err) < $(sqrt(cutoff)), [norm = $(norm(M))| normS = $(norm(vals))]")
        end
    else
        @warn "No cutoff given"
    end


#   # # TODO this can cause err ??
#   # isqZ = ITensor()
#   # try 
#   #   isqZ = ITensor(pinv(matrix(Z), atol=1e-14)^0.5, inds(Z))
#   # catch e 
#   #   @show(inds(Z))
#   #   @show(matrix(Z))
#   #   @show(pinv(matrix(Z)))
#   #   @show(pinv(matrix(Z))^0.5)
#   #   isqZ = ITensor(pinv(matrix(Z))^0.5, inds(Z))
#   # end
  
#   isqZ = ITensor(pinv(matrix(Z), atol=1e-14)^0.5, inds(Z))

#   Ot = F.Vt * isqZ 
#     norm_err = norm(a_rec-arr_a)/norm(arr_a)
#     if norm_err > 1e-6
#         @warn("AT/SVD decomp maybe not accurate, norm error $norm_err")
#     end

    return Eigen(vals, O), spec, norm_err
end




""" When called on ITensors, `symm_svd` returns a single `TruncSVD` object
""" 
function symm_svd(a::ITensor, linds; cutoff=nothing, maxdim=nothing)
    rinds = uniqueinds(a, linds)

    cL = combiner(linds)
    cR = combiner(rinds)
    am = matrix(a * cL * cR)

    # returns SVD() object and spectrum - recall unpacking of SVD() gives 
    F, spec, trunc_err = symm_svd(am; cutoff, maxdim)
    #u,s,ustar = F

    index_u = Index(size(F.S,1), "u")
    index_v = Index(size(F.S,1), "v")

    #@show size(F.Vt), dim(index_v), dim(combinedind(cR))

    u = ITensor(F.U, combinedind(cL), index_u) * dag(cL)
    s = diag_itensor(F.S, index_u, index_v)
    uT = ITensor(F.Vt, index_v, combinedind(cR)) * dag(cR)
  
    return ITensors.TruncSVD(u,s,uT, spec, index_u, index_v)
end

""" If we don't specify linds, assume we're working with a matrix and just do index1 vs index2 """
function symm_svd(a::ITensor; cutoff=nothing, maxdim=nothing)
    if ndims(a) != 2
        @error "$(ndims(a))(>2)-dimensional tensor, need to specify linds"
    end

    symm_svd(a, ind(a,1); cutoff, maxdim)

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