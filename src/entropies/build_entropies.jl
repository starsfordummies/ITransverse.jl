""" Given a vector of eigenvalues, computes the renyi `alpha` entropy from it (in a supposedly efficient way) """
function salpha(eigs::AbstractVector{<:Number}, alpha::Number)
    if alpha ≈ 0
        return log(length(eigs))
    elseif alpha ≈ 1
        return -mapreduce(λ -> λ * log(λ), +, eigs)
    else
        return log(mapreduce(λ -> λ^alpha, +, eigs)) / (1 - alpha)
    end
end

#
""" Given an input spectrum, normalizes it (unless normalize_eigs=false)
and builds the corresponding entropies (S0, S05, S1, S2, S4). """
function renyi_entropies(spectrum::AbstractVector{<:Number}; normalize_eigs::Bool=true)
    renyi_entropies([spectrum]; normalize_eigs)
end

function renyi_entropies(spectra::AbstractVector{<:AbstractVector};
        normalize_eigs::Bool = true)

    el_type = promote_type(Float64, map(eltype, spectra)...) 

    n  = length(spectra)
    S0 = Vector{el_type}(undef, n)
    S05 = Vector{el_type}(undef, n)
    S1 = Vector{el_type}(undef, n)
    S2 = Vector{el_type}(undef, n)
    S4 = Vector{el_type}(undef, n)

    for (i, eigs) in enumerate(spectra)
        eigs_cpu = Vector{el_type}(Array(eigs))
        eigs_n   = normalize_eigs ? eigs_cpu ./ sum(eigs_cpu) : eigs_cpu
        S0[i] = salpha(eigs_n, 0)
        S05[i] = salpha(eigs_n, 0.5)
        S1[i] = salpha(eigs_n, 1.0)
        S2[i] = salpha(eigs_n, 2.0)
        S4[i] = salpha(eigs_n, 4.0)

    end
    return (; S0, S05, S1, S2, S4)
end


function renyi_entropies_old(spectra::AbstractVector{<:AbstractVector};
                         which_ents = [0.5, 1, 2],
                         normalize_eigs::Bool = true)


    el_type = promote_type(Float64, map(eltype, spectra)...) 
    #allents = Dict{String, Vector{el_type}}()
    n       = length(spectra)
    allents = Dict("S$(alpha)" => Vector{el_type}(undef, n) for alpha in which_ents)

    for (i, eigs) in enumerate(spectra)
        eigs_cpu = Vector{el_type}(Array(eigs))   # GPU → CPU, no-op if already CPU
        eigs_n   = normalize_eigs ? eigs_cpu ./ sum(eigs_cpu) : eigs_cpu
        for alpha in which_ents
            allents["S$(alpha)"][i] = salpha(eigs_n, alpha)
        end
    end

    return allents
end


""" From a matrix of (Nbonds x Nvalues), takes each row and computes the VN entropy from it.
Entries with abs(λ) ≤ ε are excluded from the sum (no log called on them).
Supports complex λ, in which case the returned entropies are also complex. """
function vn_from_matrix(λ_matrix::AbstractMatrix{T}; ε::Real = 1e-14) where T
    contrib = @. ifelse(abs(λ_matrix) > ε, -λ_matrix * log(λ_matrix), zero(T))
    return vec(sum(contrib, dims=2))
end
