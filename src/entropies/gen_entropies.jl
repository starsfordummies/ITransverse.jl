



""" Given the eigenvalue spectrum of a RDM/RTM, build various entropies"""
function build_entropies(spectra::Vector)
    vns = []
    tsallis2 = []
    renyi2s = [] 
    traces_Tt2 = []
    traces_Tt12 = []
    traces_Tt14 = []
    for eigss in spectra
        vn = 0.
        r2 = 0.
        r12 = 0.
        r14 = 0.
        for n in eachindex(eigss)
            p = eigss[n]       # I don't think we need the ^2 here 
            if abs(p) > 1e-15
                vn -= p * log(p)
                r2 += p^2
                r12 += p^0.5
                r14 += p^0.25
            end
        end
        push!(vns, vn)
        push!(tsallis2, - (r2 - 1.))
        push!(renyi2s, - log(r2))
        push!(traces_Tt2, r2)
        push!(traces_Tt12, r12)
        push!(traces_Tt14, r14)

    end

    entropies = Dict(:vn=>vns, :tsallis2=>tsallis2, :renyi2=>renyi2s,
     :traces_Tt2=>traces_Tt2, :traces_Tt12=>traces_Tt12, :traces_Tt14=>traces_Tt14)
    return entropies
end

