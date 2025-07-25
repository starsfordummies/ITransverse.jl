using ITensors
using ITensorMPS 
using JLD2

using ITransverse
using ITransverse: vX, vZ, vI
#using ITransverse.ITenUtils

#ITensors.enable_debug_checks()


function main_folded_pm()

    #tp = ising_tp()
    tp =  tMPOParams(0.1, build_expH_ising_murg, IsingParams(1.0, 0.5, 0.0), 0, [1,0])

    tp = tMPOParams(tp; nbeta=0)

    b = FoldtMPOBlocks(tp)


    cutoff = 1e-12
    maxbondim = 80
    itermax = 500
    eps_converged=1e-8

    truncp = TruncParams(cutoff, maxbondim, "left")  # "left" = start from initial state (Hastings)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_R")

    #sigX = ComplexF64[0,1,1,0]
    #sigZ = ComplexF64[1,0,0,-1]

    evs = [] 

    rvecs = []
    ds2s = []
    r2s = [] 


    ts = 40:10:40
    alltimes = ts.* tp.dt

    infos = Dict(:tp => tp, :pm_params => pm_params, :b => b, :times => alltimes)


    for Nsteps in ts

        time_sites = siteinds(4, Nsteps)

        init_mps = folded_right_tMPS(b, time_sites)

        mpo_1 = folded_tMPO(b, time_sites)

        # @show ev
        ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_1, pm_params) 

        ev = compute_expvals(ll, rr, ["X","Z"], b)
        @show ev

        push!(rvecs, rr)
        push!(evs, ev)
        push!(ds2s, ds2_pm)

        push!(r2s, ITransverse.rtm2_contracted_normalized(ll, rr))
    end

    return rvecs, evs, ds2s, r2s, ts, infos
end



rvecs, evs, ds2s, r2s, alltimes, infos = main_folded_pm()

println(evs)

