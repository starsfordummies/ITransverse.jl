using ITensors, ITensorMPS, JLD2
using ITransverse.ITenUtils
#using Plots

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state

function main_cone()

    JXX = 1.0  
    hz = 1.5 # 1.05
    gx = 0.0 # 0.5

    dt = 0.1

    nbeta = 0

    #up_state = Vector{ComplexF64}([1,0])
    #plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    init_state = up_state

    cutoff = 1e-10
    maxbondim = 512
    direction = "left"

    optimize_op = vI
    
    truncp = TruncParams(cutoff, maxbondim, direction)

    Nsteps = 16

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state)

    c0, b = init_cone(tp)

    cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, which_evs=["X","Z"], which_ents=["VN","GENVN","GENR2","GENR2_Pz","GENVN_Pz"], checkpoint=100)

    psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)

    return  psi, psiR, chis, expvals, entropies, infos, last_cp

end

psi, psiR, chis, expvals, entropies, infos, last_cp = main_cone()

println(chis)
println(real(expvals["Z"]))
