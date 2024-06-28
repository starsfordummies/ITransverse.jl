using ITensors, ITensorMPS, JLD2
using ITransverse.ITenUtils
using Revise
using Plots

using ITransverse

function main_cone()

    JXX = 1.0  
    hz = 1.05
    gx = 0.5

    dt = 0.1

    nbeta = 0

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    optimize_op = sigZ 

    zero_state = Vector{ComplexF64}([1,0])
    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    init_state = plus_state

    cutoff = 1e-14
    maxbondim = 200
    ortho_method = "SVD"

    truncp = trunc_params(cutoff, maxbondim, ortho_method)

    Nsteps = 30

    #time_sites = siteinds("S=3/2", 1)

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params(build_expH_ising_murg, mp, nbeta, init_state, [1,0,0,1])

    # mp = model_params("S=1/2", JXX, hz, 0.0, dt)
    # tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_murg, mp, dt, nbeta, init_state)


    c0 = init_cone(tp)

    # TODO remember ev_ start at T=2dt actually (one already from init_cone)
    #c0, c0r, evs_x, evs_z, chis, overlaps, entropies= run_cone(c0, Nsteps, optimize_op, tp, truncp)

    psi, psiR, chis, expvals, entropies, infos = run_cone(c0, Nsteps, optimize_op, tp, truncp)

    jldsave("cp_cone.jld2"; psi, psiR, chis, expvals, entropies, infos)

    return  psi, psiR, chis, expvals, entropies, infos 

end

psi, psiR, chis, expvals, entropies, infos = main_cone()

#resu = ITransverse.resume_cone("cp_cone.jld2", 10)



# println(ev_x)
# println(ev_z)

# #a = jldopen("test_future/time_evolution/plus_04.jld2")

# xs = 2:length(ev_x)+1

# pl1 = scatter(xs, real(ev_z))
# scatter!(pl1, xs, real(ev_x))

# plot!(pl1, plot!(ITransverse.ITenUtils.bench_X_04_plus[1:end]))
# pl2 = plot(chis) 

# # plot!(pl1, a["Sx"])
# # plot!(pl1, a["Sz"])


# plot(pl1, pl2)
