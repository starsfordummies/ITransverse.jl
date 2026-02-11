using ITensors
using ITensorMPS

using ITransverse
using ITransverse: vX, vZ, vI
#using ITransverse.ITenUtils

#ITensors.enable_debug_checks()


function main_folded_pm()

    tp = ising_tp()
    tp =  tMPOParams(0.1, expH_ising_murg, IsingParams(1.0, 0.7, 0.0), 0, [1,0])

    tp = tMPOParams(tp; nbeta=0)


    b = FoldtMPOBlocks(tp)

    cutoff = 1e-12
    maxdim = 128
    itermax = 800
    stuck_after = 100
    eps_converged=1e-9

    truncp = TruncParams(cutoff, maxdim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_R", "overlap", true, stuck_after)
    #pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "norm", true, stuck_after)

    evs = [] 

    rvecs = []
    ds2s = []
    r2s = [] 


    ts = 50:50
    alltimes = ts.* tp.dt

    infos = Dict(:tp => tp, :pm_params => pm_params, :b => b, :times => alltimes)


    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        init_mps = folded_right_tMPS(b, time_sites)

        mpo_X = folded_tMPO(b, time_sites; fold_op=vX)
        #mpo_Z = folded_tMPO(b, time_sites; fold_op=vZ)

        mpo_1 = folded_tMPO(b, time_sites)

        ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 

        ev = compute_expvals(ll, rr, ["X","Z"], b)
        @show ev

        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)

        push!(r2s, ITransverse.gen_renyi2(ll, rr))
    end

    return rvecs, evs, ds2s, r2s, ts, infos
end



rvecs, evs, ds2s, r2s, alltimes, infos = main_folded_pm()

println(evs)
