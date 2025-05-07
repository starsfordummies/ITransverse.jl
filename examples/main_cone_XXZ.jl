using ITensors, ITensorMPS, JLD2
using ITransverse.ITenUtils

using LinearAlgebra
#using Plots

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state


local_dim = 3

J_XY = 1.0
J_ZZ = 1.25 # 1.05
hz = 0.0 # 0.5

dt = 0.05

nbeta = 0

# optimize_op = vZ
optimize_op = [diagm(ones(ComplexF64,local_dim))...]

#up_state = Vector{ComplexF64}([1,0])
#plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

up_spin1 = [1.0+0.0im, 0.0, 0.0]

up_spin_Ising = [1.0+0.0im, 0.0]

init_state = reshape(up_spin1 * up_spin1', local_dim*local_dim)

init_state_Ising = reshape(up_spin_Ising * up_spin_Ising', 2*2)


cutoff = 1e-10
maxbondim = 256
direction = "right"

truncp = TruncParams(cutoff, maxbondim, direction)

Nsteps = 10

#time_sites = siteinds("S=3/2", 1)

mp_Ising = IsingParams(1.0,1.0, 0.0)
mp = ITransverse.ChainModels.XXZParams(J_XY, J_ZZ, hz)
#tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)

# tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state, vI)

return_MPO_XXZ(p,dt) = ITransverse.ChainModels.timeEvo_MPO_2ndOrder(sites, fill("Id", 3), zeros(3), ["S+", "S-", "Sz"], [0.5*p.J_XY, 0.5*p.J_XY, p.J_ZZ], ["S-", "S+", "Sz"], ones(3), "Sz", p.hz, dt)

@show ITransverse.ChainModels.build_expH_ising_symm_svd(mp_Ising,dt)

@show return_MPO_XXZ(mp,dt)

tp_Ising = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp_Ising, nbeta, init_state_Ising, vI)

Wl, Wc, Wr = ITransverse.build_WW(tp_Ising)


tp = tMPOParams(dt, f_XXZ, mp, nbeta, init_state, optimize_op)

FoldtMPOBlocks(tp)

c0, b = init_cone(tp)



cone_params = ConeParams(;truncp, opt_method="RDM", optimize_op, which_evs=["Sx","Sz"], which_ents=["VN"], checkpoint=20)

psi, psiR, chis, expvals, entropies, infos, last_cp = run_cone(c0, b, cone_params, Nsteps)

# return  psi, psiR, chis, expvals, entropies, infos, last_cp


# psi, psiR, chis, expvals, entropies, infos, last_cp = main_cone()

println(chis)
println(real(expvals["Sz"]))
