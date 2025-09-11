using ITensors, ITensorMPS
using Observers
using Test

using ITransverse
using ITransverse: vX, plus_state

  measure_Z(; state) = expect(state, "Z"; sites=length(state) ÷ 2)
  current_time(; current_time) = current_time
  return_state(; state) = state
  obs = observer(
    "times" => current_time, "states" => return_state, "Z" => measure_Z
  )


JXX = 1.0  
hz = 0.7
gx = 0.0

dt = 0.1

nbeta = 0

optimize_op = vX

init_state = plus_state

# @testset "Testing light cone for folded tMPO" begin

cutoff = 1e-12
maxbondim = 200

truncp = TruncParams(cutoff, maxbondim)

Nsteps = 30

mp = IsingParams(JXX, hz, gx)
tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state)
b = FoldtMPOBlocks(tp)

ss = siteinds("S=1/2", 80)
psi0 = productMPS(ss, "+")
H = build_H_ising(ss,mp)


state = tdvp(
    H, -3.0im, psi0; time_step=-0.1im, cutoff=1e-12, (step_observer!)=obs, outputlevel=1
  )

c0 = init_cone(b, 10; full=false)

cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["X","Z"], checkpoint=0)
psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)
#@show expvals["Z"]

@info norm(expvals["Z"] - obs.Z[11:end])
@test norm(expvals["Z"] - obs.Z[11:end]) < 0.05
