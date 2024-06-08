using Revise
using LinearAlgebra, ITensors, JLD2, Plots
using ProgressMeter

using ITransverse
using ITransverse.ITenUtils

include("./test_pm.jl")

ITensors.enable_debug_checks()

function main()

    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    JXX = 1.0  
    hz = 0.4
    dt = 0.1

    nbeta=0

    init_state = plus_state

    SVD_cutoff = 1e-20
    maxbondim = 120
    itermax = 300
    verbose=false
    ds2_converged=1e-7

    params = pparams(JXX, hz, dt, nbeta, init_state)
    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []
    evs_x2 = []
    evs_z2 = []

    ev0s = []

    evs_xs = []
    evs_zs = []
    evs_x2s = []
    evs_z2s = []

    leftvecs = []
    ds2s = []

    ts = 2:1:10

    genVNs = [] 

    renyi2s = []
    renyi2alts = []

    for Nsteps in ts

        time_sites = siteinds("S=3/2", Nsteps)

        init_mps = ITransverse.build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

        mpo_X = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, sigX, time_sites)
        mpo_Z = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, sigZ, time_sites)
        mpo_1 = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, Id, time_sites)


        #ll, rr, lO, Or, ds2_pm, dns  = powermethod(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
        ll, rr, lO, Or, vals, deltas  = pm_new(init_mps, mpo_1, mpo_X, pm_params) # kwargs)
        #ll, rr, lO, Or, vals, deltas  = pm_svd(init_mps, mpo_1, mpo_X, pm_params) # kwargs)

        llalt, ds2_pm  = powermethod_Lonly(init_mps, mpo_1, mpo_X, pm_params) 


        # @info "Checking (l|Or)"
        # ITransverse.check_gencan_left_phipsi(ll,Or)

        # @info "Checking (lO|r)"
        # ITransverse.check_gencan_left_phipsi(lO,rr)

        # genVN = generalized_entropy(ll, Or)

        # renyi2 = generalized_renyi_entropy(ll,Or, 2)
        # renyi2alt = generalized_renyi_entropy(ll,Or, 2; normalize=true)

        # push!(genVNs, genVN)
        # push!(renyi2s, renyi2)
        # push!(renyi2alts,renyi2alt)


        # @show inner(ll, llalt)
        # @show norm(ll)
        # @show norm(llalt)
        # @show(norm(ll)^2 + norm(llalt)^2 - 2*inner(ll, llalt) )
        # #sleep(10)

        ev0 = overlap_noconj(ll, rr)
        ev0_alt = overlap_noconj(llalt,llalt)
        push!(ev0s, ev0)

        lx = apply(mpo_X, ll, alg="naive", truncate=false)
        lid = apply(mpo_1, ll, alg="naive", truncate=false)

        evx = overlap_noconj(lx, rr)  #<L|X|R>
        ev1 = overlap_noconj(lid, rr) #<L|1|R>

        evx = evx/ev1

        evx2 = overlap_noconj(ll,Or)/overlap_noconj(ll,rr)

        lxs = apply(mpo_X, llalt, alg="naive", truncate=false)
        lids = apply(mpo_1, llalt, alg="naive", truncate=false)

        evxs = overlap_noconj(lxs, llalt)
        ev1s = overlap_noconj(lids, llalt)

        evxs = evxs/ev1s

        push!(evs_x, evx)
        push!(evs_xs, evxs)
        push!(evs_x2, evx2)


    end

    evs = Dict(
        "evs0" => ev0s,
        "evs_x" => evs_x, 
        "evs_z" => evs_z,
        "evs_x2" => evs_x2, 
        "evs_z2" => evs_z2, 
        "evs_xs" => evs_xs,
        "evs_zs" => evs_zs, 
        "evs_x2s" => evs_x2s, 
        "evs_z2s" => evs_z2s)

    ents = Dict("genVNs" => genVNs, "renyi2s" => renyi2s, "renyi2alts" => renyi2alts)

    return leftvecs, evs, ents, ds2s, ts 
end



leftvecs, evs, ents, ds2s, ts= main()

# jldsave("out_renyis.jld2";  leftvecs, evs, ents, ds2s, ts)

# scatter(ts, real(evs_x),label="<X>", legend=:left)
# scatter!(ts, real(evs_z), label="<Z>")

# scatter!(ts, real(evs_x2),label="<Xn>", legend=:left)
# scatter!(ts, real(evs_z2), label="<Zn>")
# #jldsave("out_folded_pm_ising.jld2"; leftvecs, evs_z, evs_x, ds2s)


# scatter!(ts, real(evs_xs),label="<Xs>", legend=:left)
# scatter!(ts, real(evs_zs), label="<Zs>")

# scatter(ts, real(evs_x2s),label="<Xsn>", legend=:left)
# scatter!(ts, real(evs_z2s), label="<Zsn>")


#plot!(ITransverse.ITenUtils.bench_X_04_plus[1:80], label=nothing)
#plot!(ITransverse.ITenUtils.bench_Z_04_plus[1:80], label=nothing)
