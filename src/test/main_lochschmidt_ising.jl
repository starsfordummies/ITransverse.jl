include("../initialize.jl")

using Revise
using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
using LsqFit

#include("../itransverse.jl")
using ITransverse



ITensors.enable_debug_checks()

function test_los(Tstart::Int; method::String="SVD")

JXX = 1.0  
hz = 1.0

dt = 0.1

nbeta = 2

zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

init_state = plus_state


SVD_cutoff = 1e-10
maxbondim = 120
itermax = 500
verbose=false
ds2_converged = 1e-4

#params = Dict("JXX" => JXX , "hz" => hz, "dt" => dt, "nbeta" => nbeta, "init_state" => init_state)
#pm_params = Dict(:itermax => itermax, :SVD_cutoff=> SVD_cutoff, :maxbondim => maxbondim, :verbose => false)


params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged, true, false, method)

out_filename = "out_ents_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"

ll_murg_s = MPS()

ds2s = Vector{Float64}[]


Ntime_steps = Tstart
Nsteps = Ntime_steps +2*nbeta
time_sites =  addtags(siteinds("S=1/2", Nsteps; conserve_qns = false), "time")


start_mps = productMPS(time_sites,"+");

mpo_L = build_ising_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)
#ll_dmrg = dmrg(mpo_L, start_mps, nsweeps=10, ishermitian=false, eigsolve_which_eigenvalue=:LR, outputlevel=3)

ll_murg_s, ds2s_murg_s  = powermethod_sym(start_mps, mpo_L, pm_params)

leading_eig = inner(conj(ll_murg_s'), mpo_L, ll_murg_s)

# silly extra check so we can see that (LTTR) = lambda^2 (LR)
OL = apply(mpo_L, ll_murg_s,  alg="naive", truncate=false)
leading_sq = overlap_noconj(OL, OL)

normalization = overlap_noconj(ll_murg_s,ll_murg_s)
println(ds2s_murg_s)

return ll_murg_s, leading_eig, leading_sq, normalization
#,  ll_dmrg

end


allens = []
allens2 = []
norms = []
for jj = 24:1:40
    _, en, en2, norm = test_los(jj, method="SVDold")
    push!(allens, en)
    push!(allens2, en2)
    push!(norms, norm)
    _, en, en2, norm = test_los(jj, method="SVD")
    push!(allens, en)
    push!(allens2, en2)
    push!(norms, norm)
    _, en, en2, norm = test_los(jj, method="EIG")
    push!(allens, en)
    push!(allens2, en2)
    push!(norms, norm)
end


# func1(t, p) = p[1] .+ p[2]./t 
# func2(t, p) = p[1] .+ p[2]./t .+ p[3]./t.^2 

# p0_1 = [1., 1.]
# p0_2 = [1., 1., 1.]

# #fit1 = curve_fit(func1, range(20,100), log.(abs.(allens)), p0_1)
# fit2 = curve_fit(func2, range(20,100), log.(abs.(allens)), p0_2)

# scatter(log.(abs.(allens)))
# #plot!(func1(range(20,130), fit1.param))
# plot!(func2(range(20,130), fit2.param))

# println(fit2.param)
# 0.5 * π/24
