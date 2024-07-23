using Revise
using ITensors, JLD2

using ITransverse
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm()

    tp = ising_tp()

    cutoff = 1e-20
    maxbondim = 120
    itermax = 100
    verbose=false
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")

    sigX = ComplexF64[0,1,1,0]

    evs = [] 

    leftvecs = []
    ds2s = []


    tp = tmpo_params(tp; nbeta=0)

    mpim = model_params(tp.mp; dt=-im*tp.mp.dt)
    tpim = tmpo_params(tp; mp=mpim)


    b = FoldtMPOBlocks(tp)
    b_im = FoldtMPOBlocks(tpim)


    infos = Dict("tp" => tp, "pm_params" => pm_params)

    ts = 50:1:50
    alltimes = ts.* tp.dt

    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        
        init_mps = folded_right_tMPS(b, time_sites)

        mpo_X = folded_tMPO(b, b_im, time_sites, sigX)
        mpo_1 = folded_tMPO(b, b_im, time_sites)


        rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_X, pm_params) 

        ev = compute_expvals(ll, rr, ["X"], b)

        push!(evs, ev)
        push!(ds2s, ds2_pm)

    end

    return rvecs, evs, ds2s, ts, infos
end



rvecs, evs, ds2s, alltimes, infos = main_folded_pm()
