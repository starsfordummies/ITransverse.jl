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

    truncp = TruncParams(cutoff, maxbondim)

    Nsteps = 30

    #time_sites = siteinds("S=3/2", 1)

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params(build_expH_ising_murg, mp, nbeta, init_state, Id)

    # mp = model_params("S=1/2", JXX, hz, 0.0, dt)
    # tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_murg, mp, dt, nbeta, init_state)


    c0, b = init_cone(tp)

    cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["X"], checkpoint=20)

    psi, psiR, chis, expvals, entropies, infos = run_cone(c0, b, cone_params, Nsteps)

    return  psi, psiR, chis, expvals, entropies, infos 

end

psi, psiR, chis, expvals, entropies, infos = main_cone()
