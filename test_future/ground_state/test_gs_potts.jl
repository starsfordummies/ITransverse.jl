using ITensors
using Plots
using LsqFit
plotlyjs()

include("../models/potts.jl")
include("../power_method/compute_entropies.jl")


function gs_potts(N::Int, JJ::Real, ff::Real, ham_function)
    
    sites = siteinds("S=1",N)

    #H = build_H_potts(sites, JJ, ff)
    
    H = ham_function(sites, JJ, ff)


    nsweeps = 20 # number of sweeps is 5
    maxdim = [400] # gradually increase states kept
    cutoff = [1E-10] # desired truncation error

    psi0 = randomMPS(sites,2)

    e_gs, psi_gs = dmrg(H,psi0; nsweeps, maxdim, cutoff)

    return e_gs, psi_gs

end

println("Using autoMPO")
@time eGS, psiGS = gs_potts(100, 1.0, 0.8, build_H_potts)

#@time eGS2, psiGS2 = gs_potts(200, 1.0, 0.8, build_H_potts_alt)

println("Using manual")
@time eGS2, psiGS2 = gs_potts(100, 1.0, 0.8, build_H_potts_manual)

println("Using manual low tri")
@time eGS3, psiGS3 = gs_potts(100, 1.0, 0.8, build_H_potts_manual_lowtri)


println("$eGS  vs  $eGS2 vs $eGS3")
println(inner(psiGS', psiGS2))
println(inner(psiGS', psiGS3))
println(inner(psiGS2', psiGS3))


S_of_ll = vn_entanglement_entropy(psiGS)


# L = length(psiGS)
# # Fit - parameters are [1]central charge [2]offset 
# pars0 = [0.6, 0.2]
# WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

# x0 = 1:L-1
# y0 = S_of_ll

# fit_s_ofW        = curve_fit(WplusC,x0,y0,pars0)  

# xs     = minimum(x0):0.1:maximum(x0)

# yfitted = [WplusC(ll, fit_s_ofW.param) for ll in xs]
# scatter(vn_entanglement_entropy(psiGS))
# plot!(xs,yfitted)

# println(fit_s_ofW.param)
