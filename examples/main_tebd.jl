using ITensors, ITensorMPS
using ITransverse
using Plots

JXX = 1.0
hz = 1.05
gx = -0.5

dt = 0.12

zero_state = Vector{ComplexF64}([1, 0])
plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

init_state = plus_state
#init_state = zero_state

nbeta = 0

cutoff = 1e-10
maxbondim = 256

truncp=  TruncParams(cutoff, maxbondim)

ss = siteinds("S=1/2", 40)

truncp = TruncParams(cutoff, maxbondim)

mp = IsingParams(JXX, hz, gx)


tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state, init_state)
resu2 = tebd_ev(ss, tp, 40, ["X","Z"], truncp)


tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state, init_state)
resu3 = tebd_ev(ss, tp, 40, ["X","Z"], truncp)

tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, init_state)
#tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state, init_state)
resu = tebd_ev(ss, tp, 40, ["X","Z"], truncp)

p1 = plot(resu["Z"])
scatter!(p1, resu2["Z"])
scatter!(p1, resu3["Z"],marker=:x)

p2 = plot(resu["chis"])
scatter!(p2, resu2["chis"])
scatter!(p2, resu3["chis"],marker=:x)

plot(p1,p2)