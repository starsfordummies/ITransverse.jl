using ITensors, ITensorMPS
using Revise
using ITransverse
using Test

tp = ising_tp()

maxbondim=100
Ntime_steps = 60
nbeta =2 

mp = model_params("S=1/2", 1, 1, 0, 0.1)

tp = tmpo_params(tp; nbeta=2, mp=mp, tr=tp.bl)

Nsteps = nbeta + Ntime_steps + nbeta

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

#mpo_L, start_mps = build_ising_fw_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)
#psi_trunc, ds2s_murg_s = powermethod_sym(start_mps, mpo_L, pm_params)

mpo, start_mps = fw_tMPO(tp, time_sites)


mycutoff=1e-12
itermax=600
ds2_converged = 1e-6

@testset "Testing Loschmidt echo optimizers" begin


truncp = trunc_params(mycutoff, maxbondim, "SVD")
pm_params = PMParams(truncp, itermax, ds2_converged, true)

psi_svd, ds2 = powermethod_sym(start_mps, mpo, pm_params)

truncp = trunc_params(mycutoff, maxbondim, "EIG")
pm_params = PMParams(truncp, itermax, ds2_converged, true)

psi_eig, ds2 = powermethod_sym(start_mps, mpo, pm_params)

truncp = trunc_params(mycutoff, maxbondim, "RDM")
pm_params = PMParams(truncp, itermax, ds2_converged, true)

psi_rdm, ds2 = powermethod_sym(start_mps, mpo, pm_params)

    vn_svd = vn_entanglement_entropy(psi_svd)
    vn_eig = vn_entanglement_entropy(psi_eig)
    vn_rdm = vn_entanglement_entropy(psi_rdm)


    # plot(vn_svd)
    # plot!(vn_eig)
    # plot!(vn_rdm)

  @test  norm( vn_svd - vn_eig )/norm(vn_svd) < 0.01
  @test  norm( vn_svd - vn_rdm )/norm(vn_svd) < 0.01

end