#
# * Truncation related stuff 
# ! SHOULD BE REPLACED BY MY svdeig_symm.jl in IGensors
#

"""
Truncation of array of SVD 
"""
function mytruncate(svs::Array{Float64}, trunc_error::Real, chi_max::Int)

    # SVD are in decreasing order 

    # eg trunc error 0.01
    #svs = 0.9 0.05 0.02 0.009 0.007 0.001 .. 
    #                       ^k here denotes the first index we truncate
    # but if I sum 0.009+0.007 I get 0.016 > trunc_err, so I should go below 

    svs_normalized = normalize(svs)
    curr_error = zero(Float64)

    # Find the location first SV smaller than the cutoff - above this one we surely don't want to cut
    k = findfirst(x -> x < trunc_error, svs_normalized )

    if isnothing(k)  # all SVs are above the trunc_error, then I keep all of them
        cut_k = length(svs_normalized)
    else
        for ik = k:length(svs_normalized)
            #println("try $ik")
            curr_error = norm(svs_normalized[ik:end])
            if  curr_error < trunc_error
                cut_k = ik - 1
                break
            end
            # if we reached the end and the sum is still above that
            cut_k = ik 
        end
    end

    if cut_k > chi_max 
        cut_k = chi_max
        curr_error = norm(svs_normalized[cut_k:end])
    end


    # We return cut_k, the last element of SVs which we should keep
    return cut_k, curr_error
end



"""
Truncation of array of eigenvalues (can be complex)
"""
function mytruncate_eig(evs::Array{T}, trunc_error::Real, chi_max::Int) where T<:Union{ComplexF64,Float64}

    k0 = length(evs)
 
    evs_normalized = sort(normalize(evs); by=abs, rev=true)  # check this 
    curr_error = zero(Float64)

    # Find the location first eig smaller than the cutoff - above this one we surely don't want to cut
    k = findfirst(x -> abs(x) < trunc_error, evs_normalized )

    if isnothing(k)  # all evs are above the trunc_error, then I keep all of them
        cut_k = k0
    else
        for ik = k:k0
            #println("try $ik")
            curr_error = norm(evs_normalized[ik:end])
            if  curr_error < trunc_error
                cut_k = ik - 1
                break
            end
            # if we reached the end and the sum is still above that
            cut_k = ik 
        end
    end

    if cut_k > chi_max 
        cut_k = chi_max
        curr_error = norm(evs_normalized[cut_k+1:end])
    end


    # We return cut_k, the last element of SVs which we should keep
    if cut_k < k0 && curr_error > 1e-14
        println("cut $(k0-cut_k)/$k0 eigs, err $curr_error")
    end
    return cut_k, curr_error
end



"""
Performs SVD of array M, dropping (normalized) SVs below svd_cutoff and truncating at chi_max \\
Returns u,s,vd so that Mtrunc = u * s * vd
"""
function svdtrunc(M::Matrix{T}; svd_cutoff::Real, chi_max::Int) where T<:Union{Float64,ComplexF64}
#function svdtrunc(M::Matrix{T}, svd_cutoff::Real, chi_max::Int) where T<:Union{Float64,ComplexF64}


    F=SVD{T, Float64, Matrix{T}}
    try
        F=svd(M,alg=LinearAlgebra.DivideAndConquer())
    catch err
        F=svd(M,alg=LinearAlgebra.QRIteration())
    end

    k, trunc_err = mytruncate(F.S, svd_cutoff, chi_max)
    
    return F.U[:,1:k], F.S[1:k] , F.Vt[1:k,:], trunc_err
end





"""
Performs Eigenvalue decomposition of array M, sorting them by decreasing absolute value,
dropping eigs in *norm* below `cutoff` and truncating at `chi_max` \\
Returns U, D, U ^-1, trunc_err,  so that  M ≈ U * D * U⁻¹
"""
function eigtrunc(M::Array, cutoff::Real, chi_max::Int)

    F = eigen(M, sortby= x-> -abs(x))
    D= F.values
    U = F.vectors

    # TODO: should we truncate wrt cutoff or cutoff^2 ? 
    #cut2 = cutoff
    normeig2 = norm(D)^2

    D = D[abs2.(D)/normeig2 .> cutoff]

    k = min(length(D), chi_max)

    Uinv = inv(U)

    trunc_err = norm( U[:,1:k] * diagm(D[1:k]) * Uinv[1:k,:] - M)
    trunc_err_alt = norm( U[:,1:k] * diagm(D[1:k]) * pinv(U[:,1:k]) - M)

    #@show (trunc_err, trunc_err_alt)
    if trunc_err > cutoff
        @warn "Truncation error is large: $trunc_err (cutoff = $(cutoff))"
        @warn "eigs = $(F.values)"
        @warn "trunc: $D"
    end

    spec = Spectrum(F.values, trunc_err)

    return U[:,1:k], D[1:k], Uinv[1:k,:], spec
end


"""
Simple truncation made using eigenvalues for an ITensor. No assumptions on symmetry are made
"""
function eigtrunc(M::ITensor, cutoff::Real, chi_max::Int)
    
    # check that it's actually a square matrix
    @assert order(M) == 2 
    if dims(M)[1] != dims(M)[2]
        @error "Not a square matrix, $(inds(M))"
    end


    (i,j) = inds(M)

    Mm = matrix(M)

    U, D, Uinv, spec = eigtrunc(Mm, cutoff, chi_max)

    iu = Index(size(D)[1],tags="eigen")
    iv = Index(size(D)[1],tags="eigen")

    U_iten = ITensor(U, i, iu)
    D_iten = diagITensor(D, iu, iv)
    Uinv_iten = ITensor(Uinv, iv, j)

    return  ITensors.TruncEigen(D_iten, U_iten, Uinv_iten, spec, iu, iv)
    #U_iten, D_iten, Uinv_iten, trunc_err 
end






"""
Truncation a TruncEigen object: truncates the objects 
"""
function mytrunceig!(F::ITensors.TruncEigen, trunc_error::Real, chi_max::Int)

    # get the k index where we want to truncate
    kcut, err  = mytruncate_eig(F.D.tensor.storage.data, trunc_error , chi_max)

    lnew = Index(kcut,tags=tags(F.l))
    rnew = Index(kcut,tags=tags(F.r))

    Vtrunc = F.V[:,1:kcut]

    F.V = ITensor(F.V.tensor )

end
