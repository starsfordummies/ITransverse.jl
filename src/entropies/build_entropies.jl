
""" Given a vector `spectra` containing sets of eigenvalue spectra of a RDM/RTM at given cuts,
 returns a dict containing various entropies"""
function old_build_entropies(spectra::Vector)
    vns = []
    tsallis2 = []
    renyi2s = [] 
    traces_Tt2 = []
    traces_Tt12 = []
    traces_Tt14 = []
    traces_Tt = []
    for eigss in spectra
        vn = 0.
        r1 = 0.
        r2 = 0.
        r12 = 0.
        r14 = 0.
    
        for λ in eigss
            if abs(λ) > 1e-15
                vn += - λ * log(λ)
                r2 += λ^2
                r12 += λ^0.5
                r14 += λ^0.25
                r1 += λ
            end
        end
        push!(vns, vn)
        push!(tsallis2, - (r2 - 1.))
        push!(renyi2s, - log(r2))
        push!(traces_Tt, r1)
        push!(traces_Tt2, r2)
        push!(traces_Tt12, r12)
        push!(traces_Tt14, r14)

    end

    entropies = Dict(:vn=>vns, :tsallis2=>tsallis2, :renyi2=>renyi2s,
        :traces_Tt=>traces_Tt, :traces_Tt2=>traces_Tt2, :traces_Tt12=>traces_Tt12, :traces_Tt14=>traces_Tt14)
    return entropies
end

function salpha(λ::Number, α)
    Sλ = 0.0
    if α ≈ 1
        Sλ= - λ * log(λ)
    else
        Sλ = λ^α
    end
    return Sλ
end


function build_entropies(spectra::Vector, which_ents::Vector{<:Real})

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

