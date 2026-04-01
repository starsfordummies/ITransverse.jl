using ITensors, ITensorMPS
#using Plots

using ITransverse

function main_cone()

    JXX = 1.0  
    hz = 0.7
    gx = 0.0 # 0.5

    dt = 0.1

    nbeta = 0

    optimize_op = vZ
    init_state = up_state

    cutoff = 1e-10
    maxdim = 256
    direction = :right

    truncp = (;cutoff, maxdim, direction, alg="RTM")

    Nsteps = 120

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, expH_ising_symm_svd, mp, nbeta, init_state)
    b = FoldtMPOBlocks(tp)
    c0 = init_cone(b)

    #@info length(c0)

    cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)

    cp = DoCheckpoint(
        "cp_cone.jld2";
        params=Dict("tparams" => tp, "cparams" => cone_params),
        save_at=0,
        f_obs = (
            SVN = s -> vn_entanglement_entropy(s.R),
            S_SVD = s -> generalized_svd_vn_entropy(s.L, s.R),
            overlap = s -> overlap_noconj(s.L, s.R),
            expvals = s -> compute_expvals(s.L, s.R, ["Z","X"], s.b)
        ),
        f_savestate = (
            L = s -> s.L,
            R = s -> s.R,
            b = s -> s.b
        )
    )

    psi, psiR, checkpt = run_cone(c0, b, cone_params, cp, Nsteps)

    return  psi, psiR, checkpt

end

@time psi, psiR, checkpt = main_cone()
