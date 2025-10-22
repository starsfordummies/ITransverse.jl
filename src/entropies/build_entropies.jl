

""" Given an eigenvalue λ, builds the corresponding contribution to the entropy, 
either -λlog(λ) if α=1 or λ^α otherwise."""
function salpha(λ::Number, α)
    Sλ = 0.0
    if α ≈ 1
        Sλ= - λ * log(λ)
    else
        Sλ = λ^α
    end
    return Sλ
end

#
""" Given an input spectrum, normalizes it (unless normalize_eigs=false) 
and builds the corresponding entropy according to the index α"""
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
            s = 0.0
            for λ in eigs
                if abs(λ) > 1e-15
                    s += salpha(λ, alpha)
                end
            end
            push!(allents["S$(alpha)"], log.(s)/(1-alpha))
        end
    end

    return allents
end


#= 
""" Given input a sum(eigenvalues^alpha), returns renyi log(sum())/1-alpha """
function renyi(traces_alpha, alpha=2)
    return log.(traces_alpha)./(1-alpha)
end
    


""" Given input a sum(eigenvalues^alpha), returns tsallis (sum()-1)/1-alpha """
function tsallis(traces_alpha, alpha=2)
    return (traces_alpha .-1)./(1-alpha)
end
=#