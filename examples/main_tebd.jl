using ITensors, ITensorMPS
using ITransverse
using Plots

JXX = -1.0
hz = 1.05
gx = -0.5
#H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X
dt = 0.10

# zero_state = Vector{ComplexF64}([1, 0])
# plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

init_state = ITransverse.plus_state
#init_state = zero_state

nbeta = 0

cutoff = 1e-10
maxbondim = 256

truncp=  TruncParams(cutoff, maxbondim)

ss = siteinds("S=1/2", 18)

truncp = TruncParams(cutoff, maxbondim)

mp = IsingParams(JXX, hz, gx)


tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, init_state)
#tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state, init_state)
resu = tebd_ev(ss, tp, 60, ["X","Z"], truncp)


tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state, init_state)
resu2 = tebd_ev(ss, tp, 60, ["X","Z"], truncp)


tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state, init_state)
resu3 = tebd_ev(ss, tp, 60, ["X","Z"], truncp)

trange = 0.2:0.1:6.1
p1 = plot(trange, resu["Z"])
scatter!(p1,trange, resu2["Z"])
scatter!(p1,trange, resu3["Z"],marker=:x)

plot!(p1,trange, resu["X"])
scatter!(p1,trange, resu2["X"])
scatter!(p1,trange, resu3["X"],marker=:x)

p2 = plot(trange,resu["chis"])
scatter!(p2,trange, resu2["chis"])
scatter!(p2,trange, resu3["chis"],marker=:x)

plot(p1,p2)


sitesp = siteinds("S=1", 20)
mpp = PottsParams(1, 0.8)
tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_potts_murg, mpp, nbeta, init_state, init_state)

pp = plot()
truncp = TruncParams(truncp; maxbondim=64)
resu3 = tebd_ev(sitesp, tp, 60, ["Σ","τplusτdag"], truncp)
plot!(pp, resu3["τplusτdag"], label="64")
truncp = TruncParams(truncp; maxbondim=256)
resu3 = tebd_ev(sitesp, tp, 60, ["Σ","τplusτdag"], truncp)
plot!(pp, resu3["τplusτdag"], label="256")
