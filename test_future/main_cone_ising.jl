using ITensors, JLD2, Dates
using Revise
using Plots

using ITransverse

function main()

    JXX = 1.0  
    hz = 0.4

    dt = 0.1

    nbeta = 0

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    zero_state = Vector{ComplexF64}([1,0])
    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    init_state = plus_state

    SVD_cutoff = 1e-24
    maxbondim = 100
    ortho_method = "SVD"

    params = pparams(JXX, hz, dt, nbeta, init_state)
    truncp = trunc_params(SVD_cutoff, maxbondim, ortho_method)

    Nsteps = 40
    #time_sites = siteinds("S=3/2", 1)

    c0 = init_cone_ising(params)

    # TODO remember ev_ start at T=2dt actually (one already from init_cone)
    c0, c0r, ev_x, ev_z, chis, overlaps = evolve_cone(c0, Nsteps, sigZ, params, truncp)

    return c0, ev_x, ev_z, chis, overlaps

end

c0, ev_x, ev_z, chis, overlaps = main()

println(ev_x)
println(ev_z)

#a = jldopen("test_future/time_evolution/plus_04.jld2")

xs = 2:length(ev_x)+1

pl1 = scatter(xs, real(ev_z))
scatter!(pl1, xs, real(ev_x))

pl2 = plot(chis) 

# plot!(pl1, a["Sx"])
# plot!(pl1, a["Sz"])


plot(pl1, pl2)
