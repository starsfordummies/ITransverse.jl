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

cutoff = 1e-12
maxdim = 128
direction = :right

Nsteps = 50

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

truncp = (; cutoff, maxdim, direction=:right, alg="densitymatrix")
 @info truncp

cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rdm = cp.obs_hist[:X][end]

@test abs(ex_rdm- ITransverse.BenchData.bench_X_04_plus[length(psi)]) < 0.001

truncp = (; cutoff, maxdim, direction=:right, alg="naiveRTM")
 @info truncp

cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lrn = cp.obs_hist[:X][end]

@test abs(ex_rtm_lrn - ex_rdm) < 0.02


truncp = (; cutoff, maxdim, direction=:left, alg="naiveRTM")
 @info truncp

cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lrnl = cp.obs_hist[:X][end]

@test abs(ex_rtm_lrnl - ex_rdm) < 0.001



truncp = (; cutoff, maxdim, direction=:right, alg="RTM")
 @info truncp

cone_params = ConeParams(;truncp, opt_method=:sym, optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lr = cp.obs_hist[:X][end]


truncp = (; cutoff, maxdim, direction=:right, alg="RTM")
 @info "(NS)", truncp

cone_params = ConeParams(;truncp, opt_method=:ns, optimize_op)
psi, psiR, cp = run_cone(c0, b, cone_params, cp, Nsteps)
ex_rtm_lrns = cp.obs_hist[:X][end]

@test abs(ex_rtm_lrns - ex_rdm) < 0.001

end
