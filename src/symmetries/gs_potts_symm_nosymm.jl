using ITensors
using Plots
using LsqFit
plotlyjs()

include("../models/potts.jl")
include("../power_method/compute_entropies.jl")
include("../ground_state/iten_dmrg_entropy_observer.jl")


function gs_potts(N::Int, ff::Real, use_symmetric::Bool)

    sites = siteinds("S1_Z3",N, conserve_qns=use_symmetric)

    H = build_H_potts_alt(sites, 1.0, ff)


    nsweeps = 30 # number of sweeps is 5
    maxdim = [50,100,200,300,400] # gradually increase states kept
    cutoff = [1E-5,1E-8,1E-10] # desired truncation error

    #if symmetric
        #state = [isodd(n) ? "Up" : "Dn" for n=1:N]
        state = ["Up" for n=1:N]
        psi0 = productMPS(sites,state)
    #else
    #    psi0 = randomMPS(sites,2)
    #end



    obs = EntropyObserver(1e-10)

    #energy,psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)


    println("Starting DMRG, symmetric = $use_symmetric")
    energy, psi = dmrg(H,psi0; nsweeps, cutoff, maxdim, observer=obs, outputlevel=0)


    if abs(ff - 1) < 1e-10
        e0_potts = -(4/3 + 2*sqrt(3)/pi)*N
        println("should be $e0_potts ")
    end

    return energy, psi
end



@time eGS, psiGS = gs_potts(120, 1.0, false)

@time eGS_s, psiGS_s = gs_potts(120, 1.0, true)


ccharge = 0.8
L = length(psiGS)

#Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
#plot!(Wl .+0.3)

S_of_ll = vn_entanglement_entropy(psiGS)

S_of_ll_s = vn_entanglement_entropy(psiGS_s)

plot(real(expect(psiGS,"ΣplusΣdag")))
scatter!(real(expect(psiGS_s, "ΣplusΣdag")))


#inner(psiGS,match_mps_indices(psiGS,psiGS_s))
    
plot(S_of_ll)
scatter!(S_of_ll_s)
# # # Fit - parameters are 1) central charge 2) offset 
# pars0 = [0.6, 0.2] 
# WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

# x0 = 1:L-1
# y0 = S_of_ll_s
# fit        = curve_fit(WplusC,x0,y0,pars0)     
# fit.param
# xbase      = minimum(x0):0.1:maximum(x0)
# yfitted = [WplusC(ll, fit.param) for ll in xbase]
# scatter(vn_entanglement_entropy(psiGS_s))
# plot!(xbase,yfitted)

