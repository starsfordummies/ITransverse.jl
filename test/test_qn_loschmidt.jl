using ITensors, ITensorMPS
using ITransverse
using Test
#using BenchmarkTools

tp = tMPOParams(0.1, ITransverse.ChainModels.build_expH_ising_murg_new, IsingParams(1, 0.7, 0), 2, [1,0])

maxbondim=256
Ntime_steps = 40
nbeta = 4

mycutoff=1e-12
itermax=600
eps_converged = 1e-6

mp = IsingParams(1, 1, 0)

tp = tMPOParams(tp; nbeta, mp=mp)

Nsteps = nbeta + Ntime_steps

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

b = FwtMPOBlocks(tp)

mpo= fw_tMPO(b, time_sites; tr = tp.bl)
start_mps = fw_tMPS(b, time_sites; LR=:right, tr = tp.bl)


truncp = TruncParams(mycutoff, maxbondim)

pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM", "norm")
psi_svd, ds2 = powermethod_sym(start_mps, mpo, pm_params)

pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "norm")
psi_rdm, ds2 = powermethod_sym(start_mps, mpo, pm_params)


vn_svd = vn_entanglement_entropy(psi_svd)
vn_rdm = vn_entanglement_entropy(psi_rdm)


# Now with the new constructors 

ss = siteinds("S=1/2",3, conserve_szparity=true)

Utim = ITransverse.ChainModels.build_expH_ising_murg_new(ss, tp.mp, -im*0.1)
Ut = ITransverse.ChainModels.build_expH_ising_murg_new(ss, tp.mp, 0.1)

tp.bl.tensor.storage
psi_i = MPS(ss, "Up")
psi_f = MPS(ss, "Up")

Uts = [Utim, Utim, fill(Ut, 40)..., Utim, Utim]
psiL, Tm, psiR = ITransverse.construct_tMPS_tMPO(psi_i, Uts, psi_f);

# TODO this is broken with QNs
# pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM", "norm")
# psis_svd, ds2 = powermethod_sym(psiR, Tm, pm_params)

pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "norm")
psis_rdm, ds2 = powermethod_sym(psiR, Tm, pm_params)

vn_rdms = vn_entanglement_entropy(psis_rdm)

@test vn_rdm ≈ vn_rdms