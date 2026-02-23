""" Given an eigenvalue λ, builds the corresponding contribution to the entropy, 
either -λlog(λ) if α=1 or λ^α otherwise."""
function salpha(λ::Number, alpha::Number)
    Sλ = 0.0
    if alpha ≈ 1
        Sλ= - λ * log(λ)
    else
        Sλ = λ^alpha
    end
    return Sλ
end

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

#
""" Given an input spectrum, normalizes it (unless normalize_eigs=false) 
and builds the corresponding entropy according to the alphas in `which_ents` """
function renyi_entropies(spectrum::Vector{<:Number}; which_ents, normalize_eigs::Bool=true)
    renyi_entropies([spectrum]; which_ents, normalize_eigs)
end


function renyi_entropies(spectra::Vector{<:AbstractVector};
                         which_ents::Vector = [0.5, 1, 2],
                         normalize_eigs::Bool = true)


    el_type = promote_type(Float64, map(eltype, spectra)...) 
    allents = Dict{String, Vector{el_type}}()

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

    return allents
end
