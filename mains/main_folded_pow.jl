using Revise
using ITensors, JLD2

using ITransverse
using ITransverse.ITenUtils

ITensors.enable_debug_checks()


function main_folded_pm()

    tp = ising_tp()

    SVD_cutoff = 1e-14
    maxbondim = 120
    itermax = 100
    verbose=false
    ds2_converged=1e-6

    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    leftvecs = []
    ds2s = []

    ts = 30:1:30

    genVNs = [] 

    renyi2s = []
    renyi2alts = []

    for Nsteps in ts

        time_sites = siteinds("S=3/2", Nsteps)

        
        init_mps = build_folded_left_tMPS(tp, time_sites)
        mpo_X = build_folded_tMPO(tp, sigX, time_sites)
        mpo_Z = build_folded_tMPO(tp, sigZ, time_sites)
        mpo_1 = build_folded_tMPO(tp, Id, time_sites)

        ll, rr, lO, Or, ds2_pm, dns  = powermethod(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
        #ll, rr, lO, Or, vals, deltas  = pm_all(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
        #ll, rr, lO, Or, vals, deltas  = pm_svd(init_mps, mpo_1, mpo_X, pm_params) # kwargs)


        llalt, ds2_pm  = powermethod_Lonly(init_mps, mpo_1, mpo_X, pm_params) 

        ev0 = overlap_noconj(ll, rr)
        push!(ev0s, ev0)


    end

    return leftvecs, evs, ents, ds2s, ts 
end



leftvecs, evs, ents, ds2s, ts= main_folded_pm()
