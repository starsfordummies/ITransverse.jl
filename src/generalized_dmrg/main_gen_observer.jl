using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
using Infiltrator
using KrylovKit

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


mutable struct ExcitedStatesObserver <: AbstractObserver
    e0s::Array{ComplexF64}
    e1s::Array{ComplexF64}
    e2s::Array{ComplexF64}
    e3s::Array{ComplexF64}

 
    #EntropyObserver(ds2_tol=0.0,len_ents=1) = new(ds2_tol, ones(len_ents))
    #ExcitedStatesObserver() = new(Array{ComplexF64}())

end


function ITensors.measure!(o::ExcitedStatesObserver; kwargs...)
    energy = kwargs[:energy]
    b = kwargs[:bond]
    psi = kwargs[:psi]
    phi = kwargs[:phi]
    PH = kwargs[:projected_operator]

    if b == round(Int,length(psi)/2)
        vals, vecs, infoKrylov = KrylovKit.eigsolve(
            PH,
            phi,
            4,
            :LM;
            ishermitian=false,
            tol=1e-14,
            krylovdim=10,
            maxiter=200,
            verbosity=1
        )

        println("observer vals")
        println(vals)

        push!(o.e0s, vals[1])
        try
            push!(o.e1s, vals[2])
        catch
            push!(o.e1s, NaN)
        end
        try
            push!(o.e2s, vals[3])
        catch
            push!(o.e2s, NaN)
        end
        try
            push!(o.e3s, vals[4])
        catch
            push!(o.e3s, NaN)
        end
    end

end


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

# nsweeps = 5
# cutoff=[1e-14] # [1e-8,1e-12,1e-15]
# maxdim = [40,80,100]


sw= Sweeps(60)
setmaxdim!(sw, 20,40,50,100)
setcutoff!(sw, 1e-6,1e-8,1e-10,1e-12)

params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, true)

out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"


eso = ExcitedStatesObserver([],[],[],[])
eso_gen = ExcitedStatesObserver([],[],[],[])


ll_murgs = Vector{MPS}()
ll_murg_s = MPS()

ds2s = Vector{Float64}[]

Tstart= 50

Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")


start_mps = productMPS(time_sites,"+");
start_mps = randomMPS(time_sites,4)

mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)

# println("#### GENERALIZED DMRG ")
# en, ll_dmrg = dmrg_gen(mpo_L, start_mps, 
#     nsweeps=nsweeps, 
#     cutoff=cutoff, maxdim=maxdim, 
#     eigsolve_verbosity=2,
#     which_decomp="eigen",
#     eigsolve_which_eigenvalue=:LM, ishermitian=false, eigsolve_maxiter=8, eigsolve_krylovdim=8, outputlevel=1)

# en_bracket = inner(dag(ll_dmrg)',mpo_L,ll_dmrg)/inner(dag(ll_dmrg),ll_dmrg)

# pj = ProjMPO(mpo_L)
# pj = position_gen!(pj, ll_dmrg, 5)
# bij = ll_dmrg[5] * ll_dmrg[6]
# temp = product(pj, bij)
# en_LRenvs =  scalar(product(bij,temp))

println("#### GENERALIZED DMRG + METHOD ONE ")
en2, ll_dmrg2 = dmrg_gen(mpo_L, start_mps, 
    sw, observer=eso_gen,
#    cutoff=cutoff, maxdim=maxdim,
    which_decomp="gen_one",
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    eigsolve_maxiter=40, eigsolve_krylovdim=8, outputlevel=1)

println("Standard DMRG")
en3, ll_dmrg3 = dmrg(mpo_L, start_mps, sw, 
    which_decomp="svd",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, 
    observer=eso,
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1)

ψpm, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

epm = inner(dag(ψpm)',mpo_L,ψpm)/inner(dag(ψpm),ψpm)
epm_dags = inner(ψpm',mpo_L,ψpm)/inner(ψpm,ψpm)

norm_noc = inner(dag(ψpm),ψpm)
norm_c = inner(ψpm,ψpm)
# normd_noc = inner(dag(ll_dmrg),ll_dmrg)
# normd_c = inner(ll_dmrg,ll_dmrg)
#println(ds2s_murg_s)



# @show(en, en_bracket, en_LRenvs)
# @show(en2, en3, epm, epm_dags)
# println("")
# @show(norm_c, norm_noc, normd_c, normd_noc)
# println("")
# @show( maxlinkdim(ll_dmrg), maxlinkdim(ll_dmrg2), maxlinkdim(ll_dmrg3), maxlinkdim(ψpm))

# println("Eigencheck")
# @show norm(noprime(mpo_L * ll_dmrg) - en * ll_dmrg)
# @show norm(noprime(mpo_L * ll_dmrg2) - en2 * ll_dmrg2)
# @show norm(noprime(mpo_L * ll_dmrg3) - en3 * ll_dmrg3)
# @show norm(noprime(mpo_L * ψpm) - epm * ψpm)

return ll_dmrg2, ll_dmrg2, ll_dmrg3, ψpm, mpo_L, eso, eso_gen
end

aa,bb,cc,dd, hh, eobs, eobs_gen = test_los();



function ttest(apsi,ham)
testP = ProjMPO(ham)
testP = position!(testP,apsi,5)

testP2 = ProjMPO(ham)
testP2 = position_gen!(testP2,apsi,5)

return testP, testP2 
end

