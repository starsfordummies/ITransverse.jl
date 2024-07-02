using LinearAlgebra
using Revise

using ITensors
using ITransverse
using Test

@testset "Testing power method" begin

    tp = ising_tp()

    cutoff = 1e-20
    maxbondim = 120
    itermax = 100
    verbose=false
    ds2_converged=1e-6

    truncp = trunc_params(cutoff, maxbondim, "SVD")

    pm_params = PMParams(truncp, itermax, ds2_converged, true)

    sigX = ComplexF64[0,1,1,0]

    b = FoldtMPOBlocks(tp)

    Nsteps = 40

    time_sites = siteinds(4, Nsteps)

    
    init_mps = folded_right_tMPS(b, time_sites)

    mpo_X = folded_tMPO(b, time_sites, sigX)
    mpo_1 = folded_tMPO(b, time_sites)

    ll, rr, ds2_pm  = powermethod(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
    #ll, rr, lO, Or, vals, deltas  = pm_all(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
    #ll, rr, lO, Or, vals, deltas  = pm_svd(init_mps, mpo_1, mpo_X, pm_params) # kwargs)

    ev = compute_expvals(ll, rr, ["X"], b)

    rralt, ds2_pm  = powermethod_Ronly(init_mps, mpo_1, mpo_X, pm_params) 

    evsym = compute_expvals(rralt, rralt, ["X"], b)

    rr_te, ds2_pm  = powermethod_svd(init_mps, mpo_1, pm_params) 

    ev_te = compute_expvals(rr_te, rr_te, ["X"], b)


    ev1 = ev["X"]
    ev2 = evsym["X"]
    ev3 = ev_te["X"]
 
    @test abs(ev1 - ev2) < 1e-8
    @test abs(ev1 - ev3) < 1e-8
    @test abs(ev1 - ITransverse.ITenUtils.bench_X_04_plus[ts[end]]) < 0.001
    @test abs(ev2 - ITransverse.ITenUtils.bench_X_04_plus[ts[end]]) < 0.001
    @test abs(ev3 - ITransverse.ITenUtils.bench_X_04_plus[ts[end]]) < 0.001

end