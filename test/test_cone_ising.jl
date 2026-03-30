using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: vX, plus_state


JXX = 1.0  
hz = 0.4
gx = 0.0

dt = 0.1

nbeta = 0

optimize_op = vX

init_state = plus_state

@testset "Testing light cone for folded tMPO" begin

cutoff = 1e-20
maxdim = 200
direction = :right

truncp = (; cutoff, maxdim, direction)

Nsteps = 30

mp = IsingParams(JXX, hz, gx)
tp = tMPOParams(dt, expH_ising_murg, mp, nbeta, init_state)
b = FoldtMPOBlocks(tp)

c0 = init_cone(b)

cp = DoCheckpoint(
        "cp_cone.jld2";
        params=tp,
        save_at=0,
        f_obs = (
            X = s -> expval_LR(s.L, s.R, [0,1,1,0], s.b),
        ),
        f_savestate = (
            L = s -> s.L,
            R = s -> s.R,
            b = s -> s.b
        )
    )

cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rdm = cp.obs_hist[:X][end]

@test abs(ex_rdm- ITransverse.BenchData.bench_X_04_plus[length(psi)]) < 0.001

cone_params = ConeParams(;truncp, opt_method="RTM_LR", optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lr = cp.obs_hist[:X][end]

@test abs(ex_rtm_lr - ex_rdm) < 0.001

cone_params = ConeParams(;truncp, opt_method="RTM_LRn", optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lrn = cp.obs_hist[:X][end]

@test abs(ex_rtm_lrn - ex_rdm) < 0.001

cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_r = cp.obs_hist[:X][end]
@test abs(ex_rtm_lr - ex_rtm_r) < 0.001


cone_params = ConeParams(;truncp, opt_method="RTM_R", optimize_op, vwidth=2)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_rw = cp.obs_hist[:X][end]
@test abs(ex_rtm_rw - ex_rtm_r) < 0.001

@show ex_rdm, ex_rtm_r, ex_rtm_lr, ex_rtm_lrn, ex_rtm_rw
end
