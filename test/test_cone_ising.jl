using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: vX, plus_state


JXX = 1.0  
hz = 0.4
gx = 0.0

dt = 0.1

nbeta = 0

#sigX = ComplexF64[0,1,1,0]
#sigZ = ComplexF64[1,0,0,-1]
#Id = ComplexF64[1,0,0,1]

optimize_op = vX

init_state = plus_state

@testset "Testing light cone for folded tMPO" begin

cutoff = 1e-20
maxbondim = 200

truncp = TruncParams(cutoff, maxbondim)

Nsteps = 20

mp = IsingParams(JXX, hz, gx)
tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state)
b = FoldtMPOBlocks(tp)

c0 = init_cone(b)

cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["X","Z"], checkpoint=0)
psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)
@show expvals

@test abs(expvals["X"][end] - ITransverse.BenchData.bench_X_04_plus[length(psi)]) < 0.001
@show(expvals["X"][end], ITransverse.BenchData.bench_X_04_plus[length(psi)])

cone_params = ConeParams(;truncp, opt_method="RTM_LR", optimize_op, which_evs=["X","Z"], checkpoint=0)
_, _, _, expvals_lr, _, _, _ = run_cone(c0, b, cone_params, Nsteps)

@test abs(expvals_lr["X"][end] - expvals["X"][end]) < 0.001
@show(expvals_lr["X"][end])

cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, which_evs=["X","Z","XX"], checkpoint=0)
_, _, _, expvals_r, _, _, _ = run_cone(c0, b, cone_params, Nsteps)
@test abs(expvals_lr["X"][end] - expvals_r["X"][end]) < 1e-6
@show(expvals_r["X"][end])


cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, which_evs=["X","Z","XX"], checkpoint=0, vwidth=2)
_, _, _, expvals_r, _, _, _ = run_cone(c0, b, cone_params, Nsteps)
@test abs(expvals_lr["X"][end] - expvals_r["X"][end]) < 1e-3
@show(expvals_r["X"][end])

end
