
""" Shorthand for simple apply(alg="naive", truncate=false) """
function applyn(O::MPO, psi::AbstractMPS; kwargs...)
    replaceprime(contractn(O, psi; kwargs...),  1 => 0)
end


""" Shorthand for apply + swap indices """
function applys(O::MPO, psi::AbstractMPS; cutoff=nothing, maxdim=nothing)
    #apply(swapprime(O, 0, 1, "Site"), psi; cutoff, maxdim)
    psiO = contract(O, prime(siteinds,psi);  cutoff, maxdim)
end

""" Shorthand for apply with no truncation + swap indices """
function applyns(O::MPO, psi::AbstractMPS; kwargs...)
    #apply(swapprime(O, 0, 1, "Site"), psi, alg="naive", truncate=false)
    replaceprime(contractn(O, prime(siteinds,psi); kwargs...), 1 => 0)
end



""" If we have dangling tensors at the right edge of an MPS """
function contract_dangling!(psi::AbstractMPS)
    while length(psi) > 1 && ndims(psi[end]) == 1 && hascommoninds(psi[end], psi[end-1]) 
        psi[end-1] = psi[end] * psi[end-1]
        pop!(psi.data)
    end
end


""" Copied from ITensorMPS but adapted so that it can also extend """
function contractn(A::MPO, ψ::MPS; truncate=false, kwargs...)

    @assert length(A) >= length(ψ)

    A = sim(linkinds, A)
    ψ = sim(linkinds, ψ)

    N = max(length(A), length(ψ)) 
    n = min(length(A), length(ψ))

    ψ_out = typeof(ψ)(N)
    for j in 1:n
        ψ_out[j] = A[j] * ψ[j]
    end
    for j = n+1:N
        ψ_out[j] = A[j] 
    end

    for b in 1:(N - 1)
        Al = linkinds(A, b)  #commoninds(A[b], A[b + 1])
        ψl = linkinds(ψ, b) #   commoninds(ψ[b], ψ[b + 1])
        l = [Al..., ψl...]
        if !isempty(l)
            C = combiner(l)
            ψ_out[b] *= C
            ψ_out[b + 1] *= dag(C)
        end
    end

    contract_dangling!(ψ_out)

    if truncate
        truncate!(ψ_out; kwargs...)
    end

    return ψ_out
end
