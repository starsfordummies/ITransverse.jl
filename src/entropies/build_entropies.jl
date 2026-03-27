""" Given a vector of eigenvalues, computes the renyi `alpha` entropy from it (in a supposedly efficient way) """
function salpha(eigs::AbstractVector{<:Number}, alpha::Number)
    if alpha ≈ 1
        return -mapreduce(λ -> λ * log(λ), +, eigs)
    else
        return log(mapreduce(λ -> λ^alpha, +, eigs)) / (1 - alpha)
    end
end

#
""" Given an input spectrum, normalizes it (unless normalize_eigs=false) 
and builds the corresponding entropy according to the alphas in `which_ents` """
function renyi_entropies(spectrum::AbstractVector{<:Number}; which_ents, normalize_eigs::Bool=true)
    renyi_entropies([spectrum]; which_ents, normalize_eigs)
end


function renyi_entropies(spectra::AbstractVector{<:AbstractVector};
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


""" From a matrix of (Nbonds x Nvalues), takes each row and computes the VN entropy from it """
function vn_from_matrix(λ_matrix::AbstractMatrix{T}) where T
    #@show T 
    λ_safe = max.(λ_matrix, eps(T))
    
    # Compute -λ*log(λ) element-wise, zeroing out tiny eigenvalues
    contrib = @. ifelse(λ_matrix > eps(T), -λ_matrix * log(λ_safe), zero(T))
    
    # Sum along columns (eigenvalues) for each row (bipartition)
    return vec(sum(contrib, dims=2))
end
