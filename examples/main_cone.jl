using ITensors, ITensorMPS, JLD2
using ITransverse.ITenUtils
#using Plots

using ITransverse

function main_cone()

    JXX = 1.0  
    hz = 1.0 # 1.05
    gx = 0.5 # 0.5

    dt = 0.1

    nbeta = 0

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    optimize_op = sigZ

    up_state = Vector{ComplexF64}([1,0])
    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    init_state = plus_state

    cutoff = 1e-10
    maxbondim = 200
    direction = "left"

    truncp = TruncParams(cutoff, maxbondim, direction)

    Nsteps = 50

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state, Id)

    c0, b = init_cone(tp)

    cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["X","Z"], which_ents=["VN"], checkpoint=20)

    psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)

    return  psi, psiR, chis, expvals, entropies, infos, last_cp

end

psi, psiR, chis, expvals, entropies, infos, last_cp = main_cone()

println(chis)
println(real(expvals["Z"]))