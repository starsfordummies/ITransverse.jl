using ITensors
using Plots: plot, plot!

include("../models/xxzmodel.jl")
include("../power_method/compute_entropies.jl")


function gs_xxz(N::Int, do_symmetric::Bool)

    sites = siteinds("S=1/2",N, conserve_szparity=do_symmetric)

    H = build_H_XXZ(sites, 1.0, 0., 0.)

    nsweeps = 10 # number of sweeps is 5
    maxdim = [20,100,100,200,300,400] # gradually increase states kept
    cutoff = [1E-8,1E-10,1E-12,1E-14] # desired truncation error

    if do_symmetric
        state = [isodd(n) ? "Up" : "Dn" for n=1:N]
        psi0 = productMPS(sites,state)
    else
        psi0 = randomMPS(sites,2)
    end

    energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
    

end

@time eGS, psiGS = gs_xxz(40, false)
@time eGS_s, psiGS_s = gs_xxz(40, true)



S_of_ll = vn_entanglement_entropy(psiGS)

S_of_ll_s = vn_entanglement_entropy(psiGS_s)

    
plot(S_of_ll)
plot!(S_of_ll_s)

# ccharge = 1.0
# L = length(psiGS)

# #Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
# #plot!(Wl .+0.3)

# S_of_ll = vn_entanglement_entropy(psiGS)
# #S_of_ll = renyi_entanglement_entropy(psiGS,2)
    
# # Fit - parameters are 1) central charge 2) offset 
# pars0 = [0.6, 0.2]
# WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

# x0 = 1:L-1
# y0 = S_of_ll
# fit        = curve_fit(WplusC,x0,y0,pars0)     
# println(fit.param)
# xbase      = minimum(x0):0.1:maximum(x0)

# yfitted = [WplusC(ll, fit.param) for ll in xbase]
# plot(S_of_ll, marker=:circle, label="data")
# plot!(xbase,yfitted, label="fit")

