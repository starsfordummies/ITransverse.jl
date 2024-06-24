using LinearAlgebra
using ITensors
using ITransverse
using Test

#! TODO There is stuff to be checked here on the environments / canonical forms ... 
@testset "Testing power method" begin

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

Nsteps = 30

time_sites = siteinds("S=3/2", Nsteps)


mpo_1 = build_folded_tMPO(tp, Id, time_sites)
mpo_X = build_folded_tMPO(tp, sigX, time_sites)


init_mps = build_folded_left_tMPS(tp, time_sites)

ll, rr, lO, Or, vals, deltas  = pm_all(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
llalt, ds2_pm  = powermethod_Lonly(init_mps, mpo_1, mpo_X, pm_params) 

ev1 = expval_LR(ll, rr, sigX, tp)
ev2 = expval_LR(llalt,llalt, sigX, tp)

@test abs(ev1 - ev2) < 1e-6
@test abs(ev1 - ITransverse.ITenUtils.bench_X_04_plus[Nsteps]) < 0.002
@test abs(ev2 - ITransverse.ITenUtils.bench_X_04_plus[Nsteps]) < 0.002


ev1 = expval_LR(ll, rr, sigZ, tp)
ev2 = expval_LR(llalt,llalt, sigZ, tp)

@test abs(ev1 - ev2) < 1e-6
@test abs(ev1 - ITransverse.ITenUtils.bench_Z_04_plus[Nsteps]) < 0.002
@test abs(ev2 - ITransverse.ITenUtils.bench_Z_04_plus[Nsteps]) < 0.002

end