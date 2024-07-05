using Revise
using ITensors, JLD2

using ITransverse
#using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm()

    tp = ising_tp()

    cutoff = 1e-14
    maxbondim = 120
    itermax = 100
    verbose=false
    eps_converged=1e-6

    truncp = trunc_params(cutoff, maxbondim, "SVD")

    pm_params = PMParams(truncp, itermax, eps_converged, true, "LR")

    sigX = ComplexF64[0,1,1,0]

    evs = [] 

    leftvecs = []
    ds2s = []

    ts = 30:1:30

    b = FoldtMPOBlocks(tp)

    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        
        init_mps = folded_right_tMPS(b, time_sites)
        mpo_X = folded_tMPO(b, time_sites, sigX)
        mpo_1 = folded_tMPO(b, time_sites)

        rr, ll, ds2_pm  = powermethod(init_mps, mpo_1, mpo_X, pm_params) 

        ev = compute_expvals(ll, rr, ["X"], b)

        push!(evs, ev)

    end

    return leftvecs, evs, ds2s, ts 
end



leftvecs, evs, ds2s, ts= main_folded_pm()
