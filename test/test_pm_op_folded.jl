using LinearAlgebra

using ITensors, ITensorMPS
using ITransverse
using Test

@testset "Testing power method" begin

    tp = ising_tp()

    cutoff = 1e-20
    maxdim = 120
    itermax = 100
    verbose=false
    eps_converged=1e-10

    truncp = TruncParams(cutoff, maxdim)

    sigX = ComplexF64[0,1,1,0]

    b = FoldtMPOBlocks(tp)

    Nsteps = 50

    time_sites = siteinds(4, Nsteps)

    
    init_mps = folded_right_tMPS(b, time_sites)

    mpo_X = folded_tMPO(b, time_sites; fold_op=sigX)
    mpo_1 = folded_tMPO(b, time_sites)

    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method="RTM_LR", normalization="norm")
    ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 

    ev = compute_expvals(ll, rr, ["X"], b)
    χ_LR = maxlinkdim(ll)

    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method="RTM_R", normalization="norm")
    ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 

    evsym = compute_expvals(ll, rr, ["X"], b)
    χ_R = maxlinkdim(ll)

    truncp = TruncParams(sqrt(cutoff), maxdim)
    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method="RDM", normalization="norm")
    ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 
    ev_rdm = compute_expvals(ll, rr, ["X"], b)
    χ_RDM = maxlinkdim(ll)

    truncp = TruncParams(sqrt(cutoff), maxdim)
    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method="RDM_R", normalization="norm")
    ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 
    ev_sym_rdm = compute_expvals(ll, rr, ["X"], b)
    χ_sym_rdm = maxlinkdim(ll)

    ev1 = ev["X"]
    ev2 = evsym["X"]
    ev3 = ev_rdm["X"]
    ev4 = ev_sym_rdm["X"]

    final_time = length(ll)

    Δ_LR = abs(ev1 - ITransverse.BenchData.bench_X_04_plus[final_time])
    Δ_R  = abs(ev2 - ITransverse.BenchData.bench_X_04_plus[final_time])
    Δ_RDM =abs(ev3 - ITransverse.BenchData.bench_X_04_plus[final_time])
    Δ_RDMsym =abs(ev4 - ITransverse.BenchData.bench_X_04_plus[final_time])
 
    @test abs(ev1 - ev2) < 1e-7
    @test abs(ev1 - ev3) < 1e-3
    @test abs(ev1 - ev3) < 1e-3

    @test Δ_LR < 0.001
    @test Δ_R < 0.001
    @test Δ_RDM < 0.001
    @test Δ_RDMsym < 0.001

    @show ev1, ev2, ev3, ev4
    @show χ_LR, χ_R, χ_RDM, χ_sym_rdm
    @show Δ_LR, Δ_R, Δ_RDM, Δ_RDMsym

end