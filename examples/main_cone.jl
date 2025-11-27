using ITensors, ITensorMPS
using ITransverse.ITenUtils
#using Plots

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state

function main_cone()

    JXX = 1.0  
    hz = -1.05
    gx = 0.5 # 0.5

    dt = 0.1

    nbeta = 0

    optimize_op = vZ
    init_state = up_state


    #up_state = Vector{ComplexF64}([1,0])
    #plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    cutoff = 1e-8
    maxbondim = 128
    direction = "right"

    truncp = TruncParams(cutoff, maxbondim, direction)

    Nsteps = 80

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state)
    b = FoldtMPOBlocks(tp)
    c0 = init_cone(b)

    #@info length(c0)

    cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, 
    which_evs=["X","Z"], which_ents=["VN"], checkpoints=40:40:100)

    psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)

    return  psi, psiR, chis, expvals, entropies, infos, last_cp

end

psi, psiR, chis, expvals, entropies, infos, last_cp = main_cone()

println(chis)
println(real(expvals["Z"]))
