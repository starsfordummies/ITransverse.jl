using LinearAlgebra
using ITensors
using ITransverse
using Test


@info "Testing power method"


zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])


JXX = 1.0  
hz = 0.4
gx = 0.0
dt = 0.1

nbeta=0

init_state = plus_state

SVD_cutoff = 1e-14
maxbondim = 120
itermax = 100
verbose=false
ds2_converged=1e-6

params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

sigX = ComplexF64[0,1,1,0]
sigZ = ComplexF64[1,0,0,-1]
Id = ComplexF64[1,0,0,1]

Nsteps = 40

time_sites = siteinds("S=3/2", Nsteps)

mp = model_params("S=1/2", JXX, hz, gx, dt)
tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_parallel_field_murg, mp, nbeta, init_state)

mpo_1 = build_folded_tMPO(tp, Id, time_sites)
mpo_X = build_folded_tMPO(tp, sigX, time_sites)


# TODO update this 
init_mps = ITransverse.build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

ll, rr, lO, Or, vals, deltas  = pm_all(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
llalt, ds2_pm  = powermethod_Lonly(init_mps, mpo_1, mpo_X, pm_params) 

ev1 = expval_LR(ll, rr, sigX, tp)
ev2 = expval_LR(llalt,llalt, sigX, tp)

@test abs(ev1 - ITransverse.ITenUtils.bench_X_04_plus[Nsteps]) < 0.002
@test abs(ev2 - ITransverse.ITenUtils.bench_X_04_plus[Nsteps]) < 0.002
