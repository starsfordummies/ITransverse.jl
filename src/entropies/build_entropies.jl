# """ Given an eigenvalue λ, builds the corresponding contribution to the entropy, 
# either -λlog(λ) if α=1 or λ^α otherwise."""
# function salpha(λ::Number, alpha::Number)
#     Sλ = 0.0
#     if alpha ≈ 1
#         Sλ= - λ * log(λ)
#     else
#         Sλ = λ^alpha
#     end
#     return Sλ
# end

#= Old 
function salpha(eigs::Vector{<:Number}, alpha::Number)
     Sλ = 0.0
     if alpha ≈ 1
        for λ in eigs
            Sλ -= λ * log(λ)
        end

    else
        for λ in eigs
            Sλ += λ^alpha
        end
        Sλ = log(Sλ) / (1-alpha)
    end

    return Sλ
end

function salpha(eigs::Vector{<:Number}, alpha::Number)
    if alpha ≈ 1
        return -sum(λ -> λ * log(λ), eigs)
    else
        return log(sum(λ -> λ^alpha, eigs)) / (1 - alpha)
    end
end

=#

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

    #= 
    # Initialize result arrays
    for alpha in which_ents
        allents["S$(alpha)"] = el_type[]  # empty array for each entropy type
    end
    
    for eigs in spectra
        # Normalize if requested
        if normalize_eigs
            eigs = eigs / sum(eigs)
        end

        # Compute Renyi entropies for this spectrum
        for alpha in which_ents
            s = salpha(eigs, alpha)
            push!(allents["S$(alpha)"], s)
        end
    end

    =#
    return allents
end
