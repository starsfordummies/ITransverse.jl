using ITensors, ITensorMPS
using Revise
using Test

using ITransverse


JXX = 1.0  
hz = 0.4
gx = 0.0

dt = 0.1

nbeta = 0

sigX = ComplexF64[0,1,1,0]
sigZ = ComplexF64[1,0,0,-1]
Id = ComplexF64[1,0,0,1]

optimize_op = sigX

plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

init_state = plus_state

@testset "Testing light cone for folded tMPO" begin

cutoff = 1e-20
maxbondim = 200

truncp = TruncParams(cutoff, maxbondim)

Nsteps = 50

mp = ModelParams("S=1/2", JXX, hz, gx)
tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state)


c0,b = init_cone(tp)

cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["X"], checkpoint=0)
psi, psiR, chis, expvals, entropies, infos = run_cone(c0, b, cone_params, Nsteps)

@test abs(expvals["X"][end] - ITransverse.ITenUtils.bench_X_04_plus[length(psi)]) < 0.001
@show(expvals["X"][end], ITransverse.ITenUtils.bench_X_04_plus[length(psi)])

cone_params = ConeParams(;truncp, opt_method="RTM_LR", optimize_op, which_evs=["X"], checkpoint=0)
_, _, _, expvals_lr, _, _ = run_cone(c0, b, cone_params, Nsteps)

@test abs(expvals_lr["X"][end] - expvals["X"][end]) < 0.001
@show(expvals_lr["X"][end])

cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, which_evs=["X"], checkpoint=0)
_, _, _, expvals_r, _, _ = run_cone(c0, b, cone_params, Nsteps)
@test abs(expvals_lr["X"][end] - expvals_r["X"][end]) < 1e-6
@show(expvals_r["X"][end])

end