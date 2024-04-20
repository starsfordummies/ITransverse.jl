using ITensors
using Plots
using LsqFit


using ITransverse
using ITransverse.ChainModels: build_H_potts_manual, build_H_potts_manual_lowtri, build_H_potts

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


L = 60 
J = 1.0
f = 1.0

println("Using autoMPO")
@time eGS, psiGS = gs_potts(L, J, f, build_H_potts)

#@time eGS2, psiGS2 = gs_potts(200, 1.0, 0.8, build_H_potts_alt)

println("Using manual")
@time eGS2, psiGS2 = gs_potts(L, J, f, build_H_potts_manual)

println("Using manual low tri")
@time eGS3, psiGS3 = gs_potts(L, J, f, build_H_potts_manual_lowtri)


println("$eGS  vs  $eGS2 vs $eGS3")
println(inner(psiGS, psiGS2))
println(inner(psiGS, psiGS3))
println(inner(psiGS2, psiGS3))


S_of_ll = vn_entanglement_entropy(psiGS)


# L = length(psiGS)
# # Fit - parameters are [1]central charge [2]offset 
# pars0 = [0.6, 0.2]
# WplusC(ll, pars) = pars[1]/6. * log.(L/pi * sin.(pi.*ll./L)) .+ pars[2] # fit func

# x0 = 1:L-1
# y0 = S_of_ll

# fit_s_ofW        = curve_fit(WplusC,x0,y0,pars0)  

# xs     = minimum(x0):maximum(x0)

# yfitted = [WplusC(ll, fit_s_ofW.param) for ll in xs]
# scatter(vn_entanglement_entropy(psiGS))
# plot!(xs,yfitted, label="fit")

# @show fit_s_ofW.param


#ccharge = 0.5
L = length(psiGS)
ls = Array(0.:1:L-2)
lsh = Array(0.5:1:L-1)

ls ./= L 
lsh ./= L 

#Wl = [ccharge/6. * log(L/pi * sin(pi*ll/L)) for ll in 1:L]
#plot!(Wl .+0.3)

scatter(ls, vn_entanglement_entropy(psiGS), label="dmrg")

# Fit - parameters are 1) central charge 2) offset 
pars0 = [0.6, 0.2]
WplusC(ll, pars) = pars[1]/6. * log.(sin.(pi.*ll)) .+ pars[2] # fit func

fit = curve_fit(WplusC,lsh,S_of_ll,pars0)     

@info "CC fit: $(fit.param[1])"

yfitted = [WplusC(ll, fit.param) for ll in ls]
plot!(ls.-1/L,yfitted, label="fit")
