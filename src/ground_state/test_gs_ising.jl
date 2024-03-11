using ITensors
using Plots: plot, plot!
using LsqFit
using Revise

#plotlyjs()

include("../power_method/utils.jl")
include("../models/ising.jl")
include("../power_method/compute_entropies.jl")


function gs_ising(N::Int; onesite::Bool=false)

    sites = siteinds("S=1/2",N)

    H = build_H_ising(sites, 1.0, 1.0)

    nsweeps = 10 # number of sweeps
    maxdim = [10,20,100,100,200,300,400] # gradually increase states kept
    cutoff = [1E-5,1E-8,1E-10,1E-12] # desired truncation error

    if onesite
        psi0 = randomMPS(sites,20)
        energy,psi = dmrg_onesite(H,psi0; nsweeps, maxdim, cutoff)
    else
        psi0 = randomMPS(sites,2)
        energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
    end
    
    return energy, psi

end



@time eGS, psiGS = gs_ising(5, onesite=false)
@time eGS_1s, psiGS_1s = gs_ising(5, onesite=true)


ccharge = 0.5
L = length(psiGS)

#Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
#plot!(Wl .+0.3)

S_of_ll = vn_entanglement_entropy(psiGS)

    
# Fit - parameters are 1) central charge 2) offset 
pars0 = [0.6, 0.2]
WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

x0 = 1:L-1
y0 = S_of_ll
fit        = curve_fit(WplusC,x0,y0,pars0)     
fit.param
xbase      = minimum(x0):0.1:maximum(x0)

yfitted = [WplusC(ll, fit.param) for ll in xbase]
plot!(xbase,yfitted)

plot(vn_entanglement_entropy(psiGS))
scatter!(vn_entanglement_entropy(psiGS_1s))


