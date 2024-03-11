
function trunc_simple(D::Vector, cutoff, rel::Bool=true)

    if rel
        D = D./maximum(abs.(D))
    end
    abseigsum = 0.
    for jj in reverse(eachindex(D))
        abseigsum += abs(D[jj])
        if abseigsum > cutoff
            return jj
        end
    end
    return 1
end

""" Returns U, S so that  A ≈ U * S * U^T """
function basic_sym_svd(A::Matrix; cutoff::Float64 = 1e-14, rel::Bool=true)

    #@assert norm(A - transpose(A)) < 1e-12

    deltaNorm_sym = norm(A - transpose(A))/norm(A)
    if deltaNorm_sym > 1e-10
        @warn "Matrix not symmetric? $deltaNorm_sym"
        if deltaNorm_sym > 1e-4
            @error "Very not symmetric!!"
        end
    end

    A = 0.5*(A + transpose(A))

    F = svd(A)
    k = trunc_simple(F.S, cutoff, rel)

    U = F.U[:,1:k]
    S = F.S[1:k]
    Vt = F.Vt[1:k,:]

    z = transpose(Vt * conj(U))
    @assert z ≈ transpose(z)

    uz = U * sqrt(z)

    @assert sqrt(z) ≈ transpose(sqrt(z))

    # @assert U * Diagonal(S) * Vt ≈ A
    # @assert z * Diagonal(S) ≈ Diagonal(S) * z
    # @assert uz * Diagonal(S) * transpose(uz) ≈ A
    deltaA= uz * Diagonal(S) * transpose(uz) - A
    if norm(deltaA)/norm(A) > 2*cutoff
        @warn "Warning: truncation large? k=$k , $(norm(deltaA) / norm(A))"
    end

    @assert size(uz,2) == size(S,1)

    return uz, S
end


""" Returns O, S so that  A ≈ O * S * O^T """
function basic_sym_eig(A::Matrix; cutoff::Float64 = 1e-14, rel::Bool=true)

    #@assert norm(A - transpose(A)) < 1e-12
    deltaNorm_sym = norm(A - transpose(A))/norm(A)
    if deltaNorm_sym > 1e-10
        @warn "Matrix not symmetric? $deltaNorm_sym"
        if deltaNorm_sym > 1e-4
            @error "Very not symmetric!!"
        end
    end

    A = 0.5*(A + transpose(A))

    if size(A)[1] > 1000
        @warn "diagonalizing big matrix! $(size(A))"
    end
    F = eigen(A , sortby= x -> -abs(x))
    k = trunc_simple(F.values, cutoff, rel)

    E = F.vectors[:,1:k]
    D = F.values[1:k]

    O = E * (transpose(E)*E)^(-0.5)

    # @assert U * Diagonal(S) * Vt ≈ A
    # @assert z * Diagonal(S) ≈ Diagonal(S) * z
     #@assert norm(O * Diagonal(D) * transpose(O) - A) < 2*cutoff

     @assert size(O,2) == size(D,1)

    return O, D
end
