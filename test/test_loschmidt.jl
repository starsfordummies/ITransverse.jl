using ITensors, ITensorMPS
using ITransverse
using Test

tp = ising_tp()

maxdim=100
Ntime_steps = 60
nbeta = 4

mp = IsingParams(1, 1, 0)

tp = tMPOParams(tp; nbeta, mp=mp)

Nsteps = nbeta + Ntime_steps

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

b = FwtMPOBlocks(tp)


mpo= fw_tMPO(b, time_sites; tr = tp.bl)
start_mps = fw_tMPS(b, time_sites; LR=:right, tr = tp.bl)

mycutoff=1e-12
itermax=600
eps_converged = 1e-8

@testset "Testing Loschmidt echo optimizers" begin


    truncp = TruncParams(mycutoff, maxdim)


    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM", "norm")
    psi_svd, ds2 = powermethod_sym(start_mps, mpo, pm_params)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_EIG", "norm")
    psi_eig, ds2 = powermethod_sym(start_mps, mpo, pm_params)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "norm")
    psi_rdm, ds2 = powermethod_sym(start_mps, mpo, pm_params)

    vn_svd = vn_entanglement_entropy(psi_svd)
    vn_eig = vn_entanglement_entropy(psi_eig)
    vn_rdm = vn_entanglement_entropy(psi_rdm)

    @test  norm( vn_svd - vn_eig )/norm(vn_svd) < 0.01
    @test  norm( vn_svd - vn_rdm )/norm(vn_svd) < 0.01

    # Reference value that should be correct
    vn_ref = [0.0013364823909116947, 0.01099203778039207, 0.024243744601652475, 0.037548695443409844, 0.05453770083164498, 0.07462683631711683, 0.09602323110540435, 0.11704522449812141, 0.13653583115728335, 0.15386754106627237, 0.16881061040224718, 0.18139062296264868, 0.19178118010907264, 0.20023682029751003, 0.2070541946980541, 0.21254682719258192, 0.21702348266053237, 0.22076693466841316, 0.22401508151200222, 0.22694826425683148, 0.2296855402194194, 0.23228994520418358, 0.234780150150846, 0.23714458308514977, 0.2393543928701465, 0.24137309975240598, 0.24316256104501505, 0.2446861810130708, 0.24591072171001843, 0.24680777977811574, 0.24735536917283843, 0.24753949687627652, 0.2473553506252575, 0.2468077420096634, 0.24591066447693244, 0.24468610592666568, 0.2431624724395841, 0.2413730045276045, 0.23935429940863298, 0.23714450017835223, 0.23478008563963046, 0.23228990514329856, 0.22968552841302642, 0.22694828208848264, 0.2240151274112351, 0.2207670051222929, 0.21702357368450187, 0.21254693459801333, 0.20705431380426625, 0.20023694583104523, 0.19178130612688501, 0.1813907433099018, 0.16881071965783373, 0.15386763568224981, 0.13653591021812805, 0.117045288158745, 0.0960232791126314, 0.0746268696588051, 0.05453772196111354, 0.037548707721076165, 0.024243751213826294, 0.010992040385940718, 0.001336482670177332]

    @test  norm( vn_svd - vn_ref )/norm(vn_svd) < 0.01
    
end