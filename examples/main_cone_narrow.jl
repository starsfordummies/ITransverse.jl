using ITensors, ITensorMPS
using ITransverse.ITenUtils
#using Plots

using ProgressMeter 

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state

function main_cone()

    JXX = 1.0  
    hz = 0.4 # 0.4
    gx = 0.0 # 0.0

    dt = 0.1

    nbeta = 0

    init_state = up_state

    cutoff = 1e-10
    maxbondim = 512
    direction = "left"

    optimize_op = vZ
    
    truncp = TruncParams(cutoff, maxbondim, direction)

    Nsteps = 20

    n_ext = 4

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.expH_ising_symm_svd, mp, nbeta, init_state)

    b = FoldtMPOBlocks(tp)
    c0 = ITransverse.init_cone(b)
    @info linkdims(c0)
    truncate!(c0, cutoff=1e-14)
    @info linkdims(c0)


    #RTM_R
    cone_params = ConeParams(;truncp, opt_method="RTM_SKEW", optimize_op, 
        which_evs=["X","Z"], 
        which_ents=["VN"], # ,"GENVN","GENR2"],
        checkpoints=0,
        vwidth=n_ext)


    cp = DoCheckpoint(
        "cp_cone.jld2";
        params=tp,
        save_at=10,
        observables = (
            SVN = s -> vn_entanglement_entropy(s.R),
            overlap = s -> overlap_noconj(s.L, s.R),
            ZX = s -> compute_expvals(s.L, s.R, ["Z","X"], s.b)
        ),
        latest_savers = (
            L = s -> s.L,
            R = s -> s.R,
            b = s -> s.b
        )
    )

    psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)

    return  psi, psiR, cp

end

psi, psiR, cp = main_cone()

println(infos[:times])
println(chis)
println(real(expvals["Z"]))
