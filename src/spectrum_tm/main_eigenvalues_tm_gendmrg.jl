using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
using KrylovKit: eigsolve
using JLD2
#using Infiltrator

using ITransverse

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
    sw = kwargs[:sweep]
    psi = kwargs[:psi]
    phi = kwargs[:phi]
    PH = kwargs[:projected_operator]

    println("$(sw) $(b)")
    vals, vecs, infoKrylov = eigsolve(  #KrylovKit
        PH,
        randomITensor(ComplexF64, inds(phi)),
        4,
        :LM;
        ishermitian=false,
        tol=1e-14,
        krylovdim=20,
        maxiter=100,
        verbosity=0
    )

    #println("observer vals")
    println("$(sw) $(b), $(vals) $(length(vals))")

    # if sw == 2 && b == 5 
    #     jldsave("pjmpo2.jld2"; PH, phi)
    #     println("saved")
    #     sleep(100)
    # end

    if length(vals) > 4
    push!(o.e0s, vals[1])
    push!(o.e1s, vals[2])
    push!(o.e2s, vals[3])
    push!(o.e3s, vals[4])
    end


end


function build_eigen_tm(Tstart::Int)

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
ds2_converged = 1e-5

# nsweeps = 8
# cutoff=[1e-14] # [1e-8,1e-12,1e-15]
# maxdim = [40,80,100]


sw= Sweeps(4)
setmaxdim!(sw, 50,80,100)
setcutoff!(sw, 1e-8,1e-10,1e-12)



params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, true)

out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"


ll_murgs = Vector{MPS}()
ll_murg_s = MPS()

ds2s = Vector{Float64}[]


Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")


#start_mps = productMPS(time_sites,"+");
start_mps = randomMPS(ComplexF64, time_sites, linkdims=40)

mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)


obs_dmrg = ExcitedStatesObserver([],[],[],[])
obs_gen = ExcitedStatesObserver([],[],[],[])


println("#### GENERALIZED DMRG  ")
en_gendmrg, ll_gendmrg = dmrg_gen(mpo_L, start_mps, 
    sw,
    which_decomp="gen_one",
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    normalize=true,
    eigsolve_maxiter=10, eigsolve_krylovdim=50, outputlevel=1, observer=obs_gen)

println("#### GEN ONESITE DMRG  ")
en_gendmrg_os, ll_gendmrg_os = dmrg_gen_onesite(mpo_L, start_mps, 
    sw,
    which_decomp="gen_one",
    eigsolve_verbosity=0,
    eigsolve_which_eigenvalue=:LM, ishermitian=false,
    normalize=true,
    eigsolve_maxiter=80, eigsolve_krylovdim=12, outputlevel=1)

println("Standard DMRG")
en_dmrg, ll_dmrg = dmrg(mpo_L, start_mps, sw, which_decomp="svd",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, 
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1, observer=obs_dmrg)


ψpm, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

epm = inner(dag(ψpm)',mpo_L,ψpm)/inner(dag(ψpm),ψpm)
#epm_dags = inner(ψpm',mpo_L,ψpm)/inner(ψpm,ψpm)

println("(L|L) gen DMRG: $(overlap_noconj(ll_gendmrg,ll_gendmrg))")
println("(L|L) gen1SITE DMRG: $(overlap_noconj(ll_gendmrg_os,ll_gendmrg_os))")
println("(L|L)  DMRG: $(overlap_noconj(ll_dmrg,ll_dmrg))")

println("(L|T|R) gen DMRG: $(inner(dag(ll_gendmrg)',mpo_L,ll_gendmrg)/inner(dag(ll_gendmrg),ll_gendmrg))")
println("(L|T|R) gen1SITE DMRG: $(inner(dag(ll_gendmrg_os)',mpo_L,ll_gendmrg_os)/inner(dag(ll_gendmrg_os),ll_gendmrg_os))")
println("(L|T|R) DMRG: $(inner(dag(ll_dmrg)',mpo_L,ll_dmrg)/inner(dag(ll_dmrg),ll_dmrg))")
println("(L|T|R) PM: $epm")


@info "Check ortho"
@show(check_gen_ortho(ll_dmrg))
@show(check_gen_ortho(ll_gendmrg))
@show(check_gen_ortho(ψpm))


return en_dmrg, en_gendmrg, en_gendmrg_os, epm, obs_dmrg, obs_gen
end


function run_tm_eig()
    times = []
    dmrgs = []
    gendmrgs = [] 
    gendmrgs_os = []
    pms = []

    obs_dmrg = []
    obs_gen = []

    for T = 20
        println("T = $(T)")
        push!(times, T)
        edm, edmgen, edmgen_os, epmm, obs_dmrg, obs_gen = build_eigen_tm(T)
        push!(dmrgs, edm)
        push!(gendmrgs, edmgen)
        push!(gendmrgs_os, edmgen_os)
        push!(pms, epmm)
    end

    results = [times, dmrgs, gendmrgs, gendmrgs_os, pms, obs_dmrg, obs_gen]
    return results
end

resu = run_tm_eig()



