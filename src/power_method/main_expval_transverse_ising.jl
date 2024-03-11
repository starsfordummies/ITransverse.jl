using Revise
using ITensors, JLD2, Dates
using LinearAlgebra
using Plots
using LsqFit

include("utils.jl")
include("../models/ising.jl")
include("symmetric_svd.jl")
include("../tmpo/build_tmpo.jl")
include("compute_entropies.jl")
include("../truncations/sweeps_trunc.jl")
include("power_method.jl")


ITensors.enable_debug_checks()

function test_los(nsteps::Int)

JXX = 1.0  
hz = 0.8
dt = 0.1
nbeta = 0

zero_state = Vector{ComplexF64}([1,0])
plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])
init_state = plus_state


SVD_cutoff = 1e-10
maxbondim = 120
itermax = 200
verbose=false
ds2_converged = 1e-4

params = pparams(JXX, hz, dt, nbeta, init_state)
pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

ll_murg_s = MPS()

ds2s = Vector{Float64}[]


mpo_I = build_ising_expval_tMPO(build_expH_ising_murg, JXX, hz, dt, nsteps, init_state, [1 0 ; 0 1])

ts = [siteind(mpo_I,jj) for jj in 1:length(mpo_I)]

mpo_X = build_ising_expval_tMPO(build_expH_ising_murg, JXX, hz, dt, ts, init_state, [0 1 ; 1 0])

start_mps = productMPS(ts,"+")

#ll_dmrg = dmrg(mpo_L, start_mps, nsweeps=10, ishermitian=false, eigsolve_which_eigenvalue=:LR, outputlevel=3)

ll_murg_s, ds2s_murg_s  = powermethod_fold(start_mps, mpo_I, mpo_X, pm_params)

leading_eig = inner(conj(ll_murg_s'), mpo_I, ll_murg_s)
ev_x = inner(conj(ll_murg_s'), mpo_X, ll_murg_s)/inner(conj(ll_murg_s'), mpo_I, ll_murg_s)

# silly extra check so we can see that (LTTR) = lambda^2 (LR)
OL = apply(mpo_I, ll_murg_s,  alg="naive", truncate=false)
leading_sq = overlap_noconj(OL, OL)

normalization = overlap_noconj(ll_murg_s,ll_murg_s)
println(ds2s_murg_s)

return ll_murg_s, leading_eig, leading_sq, normalization, ev_x

end


allens = []
allens2 = []
norms = []
evs_x = []
for jj = 1:60
    _, en, en2, norm, ev_x = test_los(jj)
    push!(allens, en)
    push!(allens2, en2)
    push!(norms, norm)
    push!(evs_x, ev_x)
end

scatter(real(evs_x))
