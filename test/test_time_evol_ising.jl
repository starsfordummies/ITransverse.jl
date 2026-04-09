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


  TMAX = 3 

JXX = 1.0  
hz = 0.7
gx = 0.4

dt = 0.1

nbeta = 0

optimize_op = vX

init_state = plus_state

# @testset "Testing light cone for folded tMPO" begin

cutoff = 1e-12
maxdim = 200
direction = :right
alg="densitymatrix"

truncp = (;cutoff, maxdim, direction, alg)

Nsteps = round(Int, TMAX/dt)

mp = IsingParams(JXX, hz, gx)

ss = siteinds("S=1/2", 40)


psi0 = productMPS(ss, "+")
H = build_H(ss, H_ising, mp)


state = tdvp(
    H, -TMAX*im, psi0; time_step=-dt*im, cutoff=1e-12, (step_observer!)=obs, outputlevel=1
  )


# Bench: light cone 
tp = tMPOParams(dt, expH_ising_symm_svd, mp, nbeta, init_state)
b = FoldtMPOBlocks(tp)

c0 = init_cone(b, 10; full=false)

cp = DoCheckpoint(
    "cp_cone.jld2";
    params=tp,
    f_obs = (
        Z = s -> expval_LR(s.L, s.R, [1,0,0,-1], s.b),
        tt = s -> length(s.L)*tp.dt
    ),
    f_savestate = (
        L = s -> s.L,
        R = s -> s.R,
        b = s -> s.b
    )
)

cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)

psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ez = cp.obs_hist[:Z][end-10:end]
#@test abs(ex_rtm_lr - ex_rtm_r) < 0.001
#@show expvals["Z"]

@show obs.Z 
@show ez[end-10:end]
@info norm(ez - obs.Z[end-10:end])
@test norm(ez - obs.Z[end-10:end]) < 0.05



# Fourth-order ising 
tp = tMPOParams(dt, expH_ising_murg, mp, nbeta, init_state)
Ut_4o = build_Ut(ss, tp, build_4o=true)

z_tebd_4o = ITransverse.evolve(psi0, Ut_4o, Nsteps; cutoff=truncp.cutoff, maxdim=truncp.maxdim)


@show z_tebd_4o[end-10:end]
@info norm(z_tebd_4o[end-20:end] - obs.Z[end-20:end])
@test norm(z_tebd_4o[end-20:end] - obs.Z[end-20:end]) < 1e-4