using Revise

using ITensors, ITensorMPS
using .IGensors
using Plots
using LsqFit

#plotlyjs()

using ITransverse
using ITransverse.ExtraUtils: vn_entanglement_entropy
using ITransverse.ChainModels: build_H_ising

function gs_ising(N::Int; onesite::Bool=false)

    sites = siteinds("S=1/2",N)

    H = build_H_ising(sites, 1.0, 1.0)

    nsweeps = 20 # number of sweeps
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



@time eGS, psiGS = gs_ising(80, onesite=false)
@time eGS_1s, psiGS_1s = gs_ising(80, onesite=true)


#ccharge = 0.5
L = length(psiGS)
ls = Array(0.:1:L-2)
lsh = Array(0.5:1:L-1)

ls ./= L 
lsh ./= L 

#Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
#plot!(Wl .+0.3)

scatter(ls, vn_entanglement_entropy(psiGS), label="2s dmrg")
scatter!(ls, vn_entanglement_entropy(psiGS_1s), label="1s dmrg")


S_of_ll = vn_entanglement_entropy(psiGS)

    

# Fit - parameters are 1) central charge 2) offset 
pars0 = [0.6, 0.2]
WplusC(ll, pars) = pars[1]/6. * log.(sin.(pi.*ll)) .+ pars[2] # fit func

fit = curve_fit(WplusC,lsh,S_of_ll,pars0)     

@info "CC fit: $(fit.param[1])"

yfitted = [WplusC(ll, fit.param) for ll in ls]
plot!(ls.-1/L,yfitted, label="fit")



