using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
#using Infiltrator

include("../power_method/utils.jl")
include("../models/ising.jl")
include("../truncations/symmetric_svd.jl")
include("../tmpo/build_tmpo.jl")
include("../power_method/compute_entropies.jl")
include("../truncations/sweeps_trunc.jl")
include("../power_method/power_method.jl")
#include("./gen_dmrg_iten.jl")

ITensors.disable_debug_checks()

#unicodeplots()

function test_los()

JXX = 1.0  
hz = 1.2
dt = 0.1

nbeta = 2

zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

init_state = plus_state


SVD_cutoff = 1e-10
maxbondim = 100
itermax = 1000
verbose=false
ds2_converged = 1e-5

nsweeps = 5
cutoff=[1e-14] # [1e-8,1e-12,1e-15]
maxdim = [40,80,100]



params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, true)

out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"


ll_murgs = Vector{MPS}()
ll_murg_s = MPS()

ds2s = Vector{Float64}[]

Tstart= 30

Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")


start_mps = productMPS(time_sites,"+");
start_mps = randomMPS(time_sites,20)

mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)

println("#### GENERALIZED DMRG ")
en, ll_dmrg = dmrg_gen_onesite(mpo_L, start_mps, 
    nsweeps=nsweeps, 
    cutoff=cutoff, maxdim=maxdim, 
    eigsolve_verbosity=0,
    which_decomp="gen_one",
    eigsolve_which_eigenvalue=:LM, ishermitian=false, eigsolve_maxiter=8, eigsolve_krylovdim=8, outputlevel=1)

en_bracket = inner(dag(ll_dmrg)',mpo_L,ll_dmrg)/inner(dag(ll_dmrg),ll_dmrg)

pj = ProjMPO(mpo_L)
pj = position_gen!(pj, ll_dmrg, 5)
bij = ll_dmrg[5] * ll_dmrg[6]
temp = product(pj, bij)
en_LRenvs =  scalar(product(bij,temp))

println("#### GENERALIZED DMRG  ")
en2, ll_dmrg2 = dmrg_gen(mpo_L, start_mps, 
    nsweeps=nsweeps, 
    cutoff=cutoff, maxdim=maxdim,
    which_decomp="gen_one",
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    eigsolve_maxiter=8, eigsolve_krylovdim=8, outputlevel=1)

println("Standard DMRG")
en3, ll_dmrg3 = dmrg(mpo_L, start_mps, nsweeps=3, which_decomp="svd",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, cutoff=cutoff, maxdim=maxdim, 
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1)

ψpm, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

epm = inner(dag(ψpm)',mpo_L,ψpm)/inner(dag(ψpm),ψpm)
epm_dags = inner(ψpm',mpo_L,ψpm)/inner(ψpm,ψpm)

norm_noc = inner(dag(ψpm),ψpm)
norm_c = inner(ψpm,ψpm)
normd_noc = inner(dag(ll_dmrg),ll_dmrg)
normd_c = inner(ll_dmrg,ll_dmrg)
#println(ds2s_murg_s)



@show(en, en_bracket, en_LRenvs)
@show(en2, en3, epm, epm_dags)
println("")
@show(norm_c, norm_noc, normd_c, normd_noc)
println("")
@show( maxlinkdim(ll_dmrg), maxlinkdim(ll_dmrg2), maxlinkdim(ll_dmrg3), maxlinkdim(ψpm))

println("Eigencheck")
@show norm(noprime(mpo_L * ll_dmrg) - en * ll_dmrg)
@show norm(noprime(mpo_L * ll_dmrg2) - en2 * ll_dmrg2)
@show norm(noprime(mpo_L * ll_dmrg3) - en3 * ll_dmrg3)
@show norm(noprime(mpo_L * ψpm) - epm * ψpm)

return ll_dmrg, ll_dmrg2, ll_dmrg3, ψpm, mpo_L
end



aa,bb,cc,dd, hh = test_los();




function ttest(apsi,ham)
testP = ProjMPO(ham)
testP = position!(testP,apsi,5)

testP2 = ProjMPO(ham)
testP2 = position_gen!(testP2,apsi,5)

return testP, testP2 
end

