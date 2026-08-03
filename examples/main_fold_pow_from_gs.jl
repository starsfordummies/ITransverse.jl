using ITensors, ITensorMPS
using ITransverse
using ITensors.Adapt: adapt

function main_folded_pm()

    tp = ising_tp()

    cutoff = 1e-20
    maxdim = 120
    itermax = 100
    eps_converged=1e-6

    truncp = (;cutoff, maxdim, alg="naiveRTM")

    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:nosym, normalization="norm")

    sigX = ComplexF64[0,1,1,0]

    evs = [] 

    rvecs = []
    ds2s = []



    space_sites = siteinds("S=1/2", 80)
    hisi = build_H(space_sites, H_ising, tp.mp)
    p0 = random_mps(space_sites)
    # cap the bond dimension so that the bulk tensors all have the same (uniform) bonds,
    # as required to use one of them as *the* boundary column of the transverse network
    _, gs = dmrg(hisi, p0; nsweeps=6, maxdim=8, cutoff=1e-10)

    tp.nbeta = 0


    b = FoldtMPOBlocks(tp)


    infos = Dict("tp" => tp, "pm_params" => pm_params)

    ts = 30:1:30
    alltimes = ts.* tp.dt

    # one (bulk) column of the ground state: a *non-product* initial state, which the
    # builders add as an extra site at the bottom of the temporal chain
    jmid = div(length(gs), 2)
    bl_gs = boundary_tensor(gs[jmid]; phys=siteind(gs, jmid),
                            left=linkind(gs, jmid-1), right=linkind(gs, jmid))

    # fold it *once*, so that both columns below share the same boundary bond index
    rho0_gs = fold_boundary(bl_gs; folded_dim=dim(b.iL))

    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        mpo_X = folded_tMPO(b, time_sites; rho0=rho0_gs, fold_op=sigX)
        mpo_1 = folded_tMPO(b, time_sites; rho0=rho0_gs)


        init_mps = ITransverse.folded_right_tMPS_in_murg(mpo_1)


        ll, rr, ds2_pm  = powermethod_op(init_mps; mpo_id=mpo_1, mpo_op=mpo_X, pm_params)
        #rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 

        ev = 0. #compute_expvals(ll, rr, ["X"], b)

        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)

    end

    return rvecs, evs, ds2s, ts, infos
end


rvecs, evs, ds2s, alltimes, infos = main_folded_pm()
