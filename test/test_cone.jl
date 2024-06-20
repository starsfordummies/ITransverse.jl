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

cutoff = 1e-14
maxbondim = 200
ortho_method = "SVD"

truncp = trunc_params(cutoff, maxbondim, ortho_method)

Nsteps = 40

#time_sites = siteinds("S=3/2", 1)

mp = model_params("S=1/2", JXX, hz, gx, dt)
tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_parallel_field_murg, mp, nbeta, init_state)

# mp = model_params("S=1/2", JXX, hz, 0.0, dt)
# tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_murg, mp, dt, nbeta, init_state)


c0 = init_cone(tp)

# TODO remember ev_ start at T=2dt actually (one already from init_cone)
#c0, c0r, evs_x, evs_z, chis, overlaps, entropies= run_cone(c0, Nsteps, optimize_op, tp, truncp)

psi, psiR, chis, expvals, entropies, infos = run_cone(c0, Nsteps, optimize_op, tp, truncp)

@test abs(expvals["X"][end] - ITransverse.ITenUtils.bench_X_04_plus[length(psi)]) < 0.002
