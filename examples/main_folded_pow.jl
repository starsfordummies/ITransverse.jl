using Revise
using ITensors, JLD2

using ITransverse
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm()

    tp = ising_tp()
    tp =  tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", 1.0, 0.7, 0.0), 0, [0,1], [1,0,0,1])


    cutoff = 1e-14
    maxbondim = 120
    itermax = 100
    eps_converged=1e-8

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_R")

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]

    evs = [] 

    rvecs = []
    ds2s = []


    tp = tMPOParams(tp; nbeta=0)

    tpim = tMPOParams(tp; dt=-im*tp.dt)


    b = FoldtMPOBlocks(tp)
    b_im = FoldtMPOBlocks(tpim)


    ts = 40:40
    alltimes = ts.* tp.dt

    infos = Dict(:tp => tp, :pm_params => pm_params, :b => b, :times => alltimes)


    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        init_mps = folded_right_tMPS(b, time_sites)

        mpo_X = folded_tMPO(b, b_im, time_sites, sigX)
        mpo_Z = folded_tMPO(b, b_im, time_sites, sigZ)

        mpo_1 = folded_tMPO(b, b_im, time_sites)


        rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_Z, pm_params) 

        ev = compute_expvals(ll, rr, ["Z"], b)

        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)

    end

    return rvecs, evs, ds2s, ts, infos
end



rvecs, evs, ds2s, alltimes, infos = main_folded_pm()

println(evs)