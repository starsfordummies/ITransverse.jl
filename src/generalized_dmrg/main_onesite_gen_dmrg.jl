using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
using Infiltrator


using ITransverse

ITensors.disable_debug_checks()

#unicodeplots()

function test_los()

JXX = 1.0  
hz = 1.0
dt = 0.1

nbeta = 2

zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

init_state = plus_state


SVD_cutoff = 1e-10
maxbondim = 100
itermax = 1000
verbose=false
ds2_converged = 1e-4

# nsweeps = 10
# cutoff=[1e-12,1e-14] # [1e-8,1e-12,1e-15]
# maxdim = [20,40,80]

sw_0n= Sweeps(["maxdim" "cutoff" 
50  1e-5 
100 1e-8 
100 1e-10 
100 1e-12 
120 1e-12 
120 1e-12 
120 1e-12 
120 1e-12 
120 1e-12 
120 1e-12 ])

sw= Sweeps(["maxdim" "cutoff" "noise"
50  1e-5 0
100 1e-8 0
100 1e-10 0
100 1e-12 0
120 1e-12 1e-4
120 1e-12 1e-5
120 1e-12 1e-6
120 1e-12 1e-8
120 1e-12 1e-10
120 1e-12 0])

eigsolve_verbosity = 0


params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, true)

out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"


ll_murgs = Vector{MPS}()
ll_murg_s = MPS()

ds2s = Vector{Float64}[]

Tstart= 20

Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")


#start_mps = productMPS(time_sites,"+");
start_mps = randomMPS(ComplexF64, time_sites, linkdims=30)

mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)


@info ("#### GEN 1site DMRG  ")
en_os_gendmrg, ll_os_gendmrg = dmrg_gen_onesite(mpo_L, start_mps, sw_0n, 
    normalize=true,
    #cutoff=cutoff, maxdim=maxdim, 
    eigsolve_verbosity=eigsolve_verbosity,
    which_decomp="gen_two",
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    eigsolve_maxiter=100, eigsolve_krylovdim=14)

en_bracket = inner(dag(ll_os_gendmrg)',mpo_L,ll_os_gendmrg)/norm_gen(ll_os_gendmrg)

#pj = ProjMPO(H=mpo_L, nsite=1)
pj = ProjMPO(0, length(mpo_L)+1, 1, mpo_L, Vector{ITensor}(undef, length(mpo_L)))

pj = position_gen!(pj, ll_os_gendmrg, 5)
bij = ll_os_gendmrg[5] 
temp = product(pj, bij)
en_LRenvs =  scalar(product(bij,temp))

@info ("#### GEN DMRG  ")
en_gendmrg, ll_gendmrg = dmrg_gen(mpo_L, start_mps, 
    #nsweeps=nsweeps, 
    sw,
    normalize=true,
    #cutoff=cutoff, maxdim=maxdim,
    which_decomp="gen_two",
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    eigsolve_maxiter=100, eigsolve_krylovdim=14, 
    eigsolve_verbosity=eigsolve_verbosity,
    outputlevel=1)

@info("Standard DMRG")
en_dmrg, ll_dmrg = dmrg(mpo_L, start_mps, 
    #nsweeps=nsweeps, 
    sw_0n, 
    which_decomp="svd",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, 
    #cutoff=cutoff, maxdim=maxdim, 
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1)


@info("Standard Onesite DMRG")
en_os_dmrg, ll_os_dmrg = dmrg_onesite(mpo_L, start_mps, 
    #nsweeps=nsweeps, 
    sw_0n,
 which_decomp="gen_one",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, 
    #cutoff=cutoff, maxdim=maxdim, 
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1)

@info("power method")
ψpm, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

epm = inner(dag(ψpm)',mpo_L,ψpm)/inner(dag(ψpm),ψpm)
epm_dags = inner(ψpm',mpo_L,ψpm)/inner(ψpm,ψpm)

# norm_noc = inner(dag(ψpm),ψpm)
# norm_c = inner(ψpm,ψpm)
# normd_noc = inner(dag(ll_dmrg),ll_dmrg)
# normd_c = inner(ll_dmrg,ll_dmrg)
#println(ds2s_murg_s)


@info "Energies"
 @show(en_os_gendmrg, en_bracket, en_LRenvs)
 @show en_gendmrg
 @show en_dmrg
 @show en_os_dmrg
 @show(epm, epm_dags)
# println("")
# @show(norm_c, norm_noc, normd_c, normd_noc)
# println("")
@info "Maxlinkdims"
 @show( maxlinkdim(ll_dmrg), maxlinkdim(ll_gendmrg), maxlinkdim(ll_os_dmrg), maxlinkdim(ll_os_gendmrg), maxlinkdim(ψpm))

# println("Eigencheck")
# @show norm(noprime(mpo_L * ll_dmrg) - en * ll_dmrg)
# @show norm(noprime(mpo_L * ll_dmrg2) - en2 * ll_dmrg2)
# @show norm(noprime(mpo_L * ll_dmrg3) - en3 * ll_dmrg3)
# @show norm(noprime(mpo_L * ψpm) - epm * ψpm)
@info "Norms"
@show norm_gen(ll_dmrg)
@show norm_gen(ll_os_dmrg)
@show norm_gen(ll_gendmrg)
@show norm_gen(ll_os_gendmrg)
@show norm_gen(ψpm)

return ll_dmrg, ll_os_dmrg, ll_gendmrg, ll_os_gendmrg, ψpm, mpo_L
#return ll_dmrg, ll_dmrg3, mpo_L
end

aa,bb,cc,dd, ee, hh = test_los();



function ttest(apsi,ham)
testP = ProjMPO(ham)
testP = position!(testP,apsi,5)

testP2 = ProjMPO(ham)
testP2 = position_gen!(testP2,apsi,5)

return testP, testP2 
end

