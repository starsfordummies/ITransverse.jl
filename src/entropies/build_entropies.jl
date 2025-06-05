

""" Given a number λ, builds the corresponding entropy contribution, be it -λlog(λ) if α=1 or λ^α otherwise."""
function salpha(λ::Number, α)
    Sλ = 0.0
    if α ≈ 1
        Sλ= - λ * log(λ)
    else
        Sλ = λ^α
    end
    return Sλ
end

""" Given an input spectrum, builds the corresponding entropy according to the index α"""
function build_entropies(spectra::Vector, which_ents::Vector)

    allents = Dict()

    for α in which_ents
        Sα = []
        Sα_n = []
        for eigs in spectra
            ss = 0.0
            ssn = 0.0
            sum_λ = sum(eigs)
            for λ in eigs[abs.(eigs) .> 1e-15]
               ss += salpha(λ, α)
               ssn += salpha(λ/sum_λ, α)
            end
            push!(Sα, ss)
            push!(Sα_n, ssn)

        end

        sums_l = [sum(eigs) for eigs in spectra]

        allents["S$(α)"] = Sα
        allents["S$(α)n"] = Sα_n 
        allents["sums"] = sums_l

    end

    return allents
end


""" Given input a sum(eigenvalues^alpha), returns renyi log(sum())/1-alpha """
function renyi(traces_alpha, alpha=2)
    return log.(traces_alpha)./(1-alpha)
end
    


""" Given input a sum(eigenvalues^alpha), returns tsallis (sum()-1)/1-alpha """
function tsallis(traces_alpha, alpha=2)
    return (traces_alpha .-1)./(1-alpha)
end
    