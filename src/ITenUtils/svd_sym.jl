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
        M::AbstractMatrix;
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

    #rec_M = MU * Diagonal(MS) * MV'

    #@show isapprox(rec_M, M, rtol=cutoff)
    #@show norm(rec_M - M)/norm(M)
    #@show truncerr, truncerr^0.5

    return SVD(MU,MS,MV'), spec

end

function check_mytrunc_svd(f::SVD, M::AbstractMatrix, cutoff::Float64)

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
    
    #u,s,v = svd(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)

    #@show isapprox(u*Diagonal(s)*vd, M)
    #@show isapprox(u*Diagonal(s)*conj(transpose(vd)), M)

    # z = u^* vdag 
    z = transpose(conj(u)) * transpose(v')
    #@show z   # should be at most block-diag
    #sq_z = sqrt.(Diagonal(diag(z)))
    # If z is diagonal, just invert its diag 
    diagz = Diagonal(diag(z))

    if norm(z - diagz) < 1e-8
        sq_z = diagz^0.5
    else
        sq_z = z^0.5
    end

    @show u 
    @show v
    @show z
    @show sq_z

    # sq_z should be symmetric 
    if norm(sq_z - transpose(sq_z))/norm(sq_z) > 1e-6
        #println("fff")
        @error "sqrt(z) not symmetric? "
        # @show norm(z - transpose(z))
        # @show norm(z - Diagonal(diag(z)))
        # @show norm(sq_z - transpose(sq_z)), norm(sq_z - transpose(sq_z))/norm(sq_z)
        # @show z 
    end

    #uz = u * Diagonal(transpose(sq_z))
    uz = u * (transpose(sq_z))

    M_rec = uz * Diagonal(s) * transpose(uz)

    # hacky but 
    if isnothing(cutoff)
         cutoff = eps(Float64)
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





""" New version trying to avoid having to go through trunc_svd.. """
function symm_svd(a::ITensor, linds; cutoff=nothing, maxdim=nothing)
    rinds = uniqueinds(a, linds)

    cL = combiner(linds)
    cR = combiner(rinds)

    ac = a * cL * cR

    iL = combinedind(cL)
    iR = combinedind(cR)

    ac = symmetrize(ac)

    # u * s * vd ≈ a 
    u,s,vd, spec = svd(ac, iL; cutoff, maxdim)
   
    index_u = commonind(u,s)
    index_v = commonind(vd,s)

    #@show matrix(u)
    #@show matrix(vd)
    #@show u * s * vd ≈ ac

    z = noprime(dag(u) * (vd' * delta(iL, iR')))

    #@show inds(z)
    # Z could still be block-diagonal. 
    # What is the safest way to invert it ? With SVD it doens't work so well,
    # maybe with eigenvalue decomp since it's symmetric? 
    # zvals, zvecs = eigen(z, index_u, index_v)
    # @info zvecs * zvals * dag(zvecs)' ≈ z
    # sq_z = zvecs * sqrt.(zvals) * dag(zvecs)

    # Best way is probably still to rely on Schur decomposition from Julia's matrix utils !? 
    sq_z = sqrt(z) # ITensor(sqrt(matrix(z)), inds(z))

    # TODO for GPU aware code : check if z is diagonal -> do on GPU
    # otherwise, bring back to CPU, do it here and bring back to GPU

    #@show matrix(z)
    #@show matrix(sq_z)

    u *= sq_z 
    uT = u * delta(iL, iR) 

    u *= delta(index_u, index_v)
    u *= dag(cL)
    uT *= dag(cR)
  
    return ITensors.TruncSVD(u,s,uT, spec, index_u, index_v)
end



""" When called on ITensors, `symm_svd` returns a single `TruncSVD` object
""" 
function symm_svd_old(a::ITensor, linds; cutoff=nothing, maxdim=nothing)
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

    z = dag(u) * (s*vd)' * delta(iL, iR') * delta(index_u', index_v)

    # Best way is probably still to rely on Schur decomposition from Julia's matrix utils !? 
    sq_z = ITensor(sqrt(matrix(z)), inds(z))

    uu = u * sq_z 
    uuL = uu * dag(cL)
    uuR = uu * delta(iL, iR) * dag(cR)
  
    return uuL, uuR
end
