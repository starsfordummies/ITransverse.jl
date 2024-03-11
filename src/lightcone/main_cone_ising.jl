using ITensors, JLD2, Dates
using Revise
using Plots

using ITransverse

#include("../itransverse.jl")
#using .ITransverse

#ITensors.enable_debug_checks()




function main()

JXX = 1.0  
hz = 0.5

dt = 0.1

nbeta = 0


sigX = ComplexF64[0,1,1,0]
sigZ = ComplexF64[1,0,0,-1]
Id = ComplexF64[1,0,0,1]



zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

init_state = plus_state

SVD_cutoff = 1e-10
maxbondim = 100
method = "SVD"

params = pparams(JXX, hz, dt, nbeta, init_state)
truncp = trunc_params(SVD_cutoff, maxbondim, method)

Nsteps = 100

time_sites = siteinds("S=3/2", 1)

c0 = init_cone_ising(params)


c0, c0r, ev_x, ev_z, chis, overlaps = evolve_cone(c0, Nsteps, sigZ, params, truncp)

return c0, ev_x, ev_z, chis, overlaps

end

c0, ev_x, ev_z, chis, overlaps = main()

println(ev_x)
println(ev_z)

pl1 = plot(real(ev_x))
pl2 = plot(chis) 

plot(pl1, pl2)

