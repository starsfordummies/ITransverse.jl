using ITensors, ITensorMPS
using ITransverse
using Test
#using BenchmarkTools


@testset "basic QN stuff " begin
mp = IsingParams(1, 0.7, 0)
nbeta = 4


tp = tMPOParams(0.1, expH_ising_murg, mp, nbeta, [1,0])

maxbondim=128
Ntime_steps = 30

mycutoff=1e-12
itermax=600
eps_converged = 1e-6

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

@test norm(vn_svd - vn_rdm) < 0.005

# Now with the new constructors 

ss = siteinds("S=1/2",3, conserve_szparity=true)

Utim = build_Ut(ss, expH_ising_murg, tp.mp; dt=-im*0.1)
Ut   = build_Ut(ss, expH_ising_murg, tp.mp; dt=0.1)

@test build_Ut(ss, expH_ising_murg, tp.mp; dt=-im*0.1) ≈ build_Ut(ss, expH_ising_murg, tp.mp; dt=0.1, build_imag=true)

tp.bl.tensor.storage
psi_i = MPS(ss, "Up")
psi_f = MPS(ss, "Up")

Uts = [Utim, Utim, fill(Ut, Ntime_steps)..., Utim, Utim]
psiL, Tm, psiR = ITransverse.construct_tMPS_tMPO(psi_i, Uts, psi_f);

end

# TODO powermethod_sym is broken with QNs
# pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM", "norm")
# psis_svd, ds2 = powermethod_sym(psiR, Tm, pm_params)

# pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "norm")
# psis_rdm, ds2 = powermethod_sym(psiR, Tm, pm_params)

# vn_rdms = vn_entanglement_entropy(psis_rdm)

# @test vn_rdm ≈ vn_rdms
