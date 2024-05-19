using ITensors
using Plots: plot,plot!

include("../models/ising.jl")
include("../power_method/compute_entropies.jl")


function gs_ising(N::Int, symmetric::Bool)

    sites = siteinds("S=1/2",N, conserve_szparity=symmetric)

    H = build_H_ising(sites, 1.0, 1.0)


    nsweeps = 5 # number of sweeps is 5
    maxdim = [20,100,100,200,300,400] # gradually increase states kept
    cutoff = [1E-5,1E-8,1E-10,1E-12] # desired truncation error

    if symmetric
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
        psi0 = productMPS(sites,state)
    else
        psi0 = random_mps(sites,2)
    end

    energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)

end


@time eGS, psiGS = gs_ising(100, false)

@time eGS_s, psiGS_s = gs_ising(100, true)


ccharge = 0.5
L = length(psiGS)

#Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
#plot!(Wl .+0.3)

S_of_ll = vn_entanglement_entropy(psiGS)

S_of_ll_s = vn_entanglement_entropy(psiGS_s)

    
plot(S_of_ll)
plot!(S_of_ll_s)


# # Fit - parameters are 1) central charge 2) offset 
# pars0 = [0.6, 0.2]
# WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

# x0 = 1:L-1
# y0 = S_of_ll
# fit        = curve_fit(WplusC,x0,y0,pars0)     
# fit.param
# xbase      = minimum(x0):0.1:maximum(x0)

# yfitted = [WplusC(ll, fit.param) for ll in xbase]
# scatter(vn_entanglement_entropy(psiGS))
# plot!(xbase,yfitted)

