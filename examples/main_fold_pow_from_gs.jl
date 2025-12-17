using ITensors, ITensorMPS
using ITransverse
using ITensors.Adapt: adapt

function main_folded_pm()

    tp = ising_tp()

    cutoff = 1e-20
    maxbondim = 120
    itermax = 100
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR", "norm")

    sigX = ComplexF64[0,1,1,0]

    evs = [] 

    rvecs = []
    ds2s = []



    space_sites = siteinds("S=1/2", 80)
    hisi = build_H(H_ising, space_sites, mp)
    p0 = random_mps(space_sites)
    _, gs = dmrg(hisi, p0, nsweeps=6)

    #gs1 = random_itensor(ComplexF64,Index(3,"lef"),  Index(3, "rig"), Index(2, "phys"))

    gs1 = adapt(ComplexF64, gs[40])

    tp = tMPOParams(tp; nbeta=0)


    b = FoldtMPOBlocks(tp)


    infos = Dict("tp" => tp, "pm_params" => pm_params)

    ts = 30:1:30
    alltimes = ts.* tp.dt

    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)
        pushfirst!(time_sites, Index(dim(b.rho0,1),tags="rho0"))

        mpo_X = folded_tMPO_in(b, time_sites; init_tensor=gs[4], init_physidx=siteind(gs,4), fold_op=sigX)
        mpo_1 = folded_tMPO_in(b, time_sites; init_tensor=gs[4], init_physidx=siteind(gs,4))

        
        init_mps = ITransverse.folded_right_tMPS_in_murg(mpo_1)


        rr, ll, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 
        #rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 

        ev = 0. #compute_expvals(ll, rr, ["X"], b)

        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)

    end

    return rvecs, evs, ds2s, ts, infos
end


rvecs, evs, ds2s, alltimes, infos = main_folded_pm()
