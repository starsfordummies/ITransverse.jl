
""" Given a vector `spectra` containing sets of eigenvalue spectra of a RDM/RTM at given cuts,
 returns a dict containing various entropies"""
function build_entropies(spectra::Vector)
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

