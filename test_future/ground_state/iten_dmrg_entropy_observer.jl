using ITensors
using Plots: plot, plot!
include("../power_method/utils.jl")
include("../models/ising.jl")
include("../models/potts.jl")
include("../models/xxzmodel.jl")

include("../power_method/compute_entropies.jl")

mutable struct EntropyObserver <: AbstractObserver
    ds2_tol::Float64
    last_vn_ent::Vector{Float64}

    #EntropyObserver(ds2_tol=0.0,len_ents=1) = new(ds2_tol, ones(len_ents))
    EntropyObserver(ds2_tol=0.0) = new(ds2_tol)

end

function ITensors.checkdone!(o::EntropyObserver;kwargs...)
sweep = kwargs[:sweep]
energy = kwargs[:energy]
psi = kwargs[:psi]

    vn_ent = vn_entanglement_entropy(psi)

    if sweep == 1
    plot(vn_ent, label="$sweep") |> display
    else
    if sweep % 5 == 0  # Clear the plot
        plot(o.last_vn_ent,label = "$(sweep-1)") |> display
        plot!(vn_ent,label = "$sweep") |> display

    else
        plot!(vn_ent,label = "$sweep") |> display
    end

    if norm(vn_ent - o.last_vn_ent) < o.ds2_tol
        println("[Sweep $sweep] [chi=$(maxlinkdim(psi))] Converged $(norm(vn_ent - o.last_vn_ent))" )
        if sweep > 2  # otherwise too early and maybe suspicious (?)
        return true
        end
    else
        println("[Sweep $sweep] [chi=$(maxlinkdim(psi))] Diff $(norm(vn_ent - o.last_vn_ent))" )
    end
    
    end

    o.last_vn_ent = vn_ent
    
    return false
end

# function ITensors.measure!(o::EntropyObserver; kwargs...)
#     energy = kwargs[:energy]
#     sweep = kwargs[:sweep]
#     bond = kwargs[:bond]
#     psi = kwargs[:psi]
#     sweep_is_done = kwargs[:sweep_is_done]

#     outputlevel = kwargs[:outputlevel]

# end



function main_gs_ising(N, gg, etol=1E-4)

    s = siteinds("S=1/2",N)

    H = build_H_ising(s, 1, gg)
    psi0 = random_mps(s,4)

    nsweeps = 30
    cutoff = 1E-8
    maxdim = [10,20,100]

    #obs = EntropyObserver(etol,length(s)-1)
    obs = EntropyObserver(etol)

    println("Starting DMRG")
    energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=0)

    return energy, psi

end



function main_gs_potts(N; ff = 1.0, etol = 1E-4)
 

    s = siteinds("S=1",N)

    H = build_H_potts(s, 1, ff)
    psi0 = random_mps(s,4)

    nsweeps = 30
    cutoff = 1E-8
    maxdim = [10,20,100]

    obs = EntropyObserver(etol)

    println("Starting DMRG")
    energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=0)

    return energy, psi
end



function main_gs_XX(N ; hh = 1.0, use_symmetries=false, etol = 1E-4 )


    s = siteinds("S=1/2",N, conserve_qns=use_symmetries) #conserve_szparity=use_symmetries)

    H = build_H_XX(s, 1, hh)

    #if use_symmetries
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
        psi0 = productMPS(s,state)
    #else
    #    psi0 = random_mps(s,4)
    #end

    nsweeps = 30
    cutoff = [1E-5,1e-6,1e-8,1e-10,1e-12]
    maxdim = [10,20,100,200,300]

    obs = EntropyObserver(etol)

    println("Starting DMRG")
    energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=0)

    return energy, psi
end



function main_gs_XXZ(N; use_symmetries=false,  etol = 1E-4)


    s = siteinds("S=1/2",N, conserve_qns=use_symmetries) #conserve_szparity=use_symmetries)

    #H = build_H_XXZ(s, -1, 0, 0)
    H = build_H_XXZ_SpSm(s, 1, 0, 0.8)

    # if use_symmetries
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
        #state = ["Up" for n=1:N]

        psi0 = productMPS(s,state)
    # else
    #     psi0 = random_mps(s,4)
    # end

    nsweeps = 20
    cutoff = [1E-5,1e-6,1e-8,1e-10,1e-12]
    maxdim = [10,20,100,200,300]

    obs = EntropyObserver(etol)

    println("Starting DMRG")
    energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=0)

    return energy, psi 
end
