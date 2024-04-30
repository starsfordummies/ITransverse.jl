
using Revise
using LinearAlgebra, ITensors, JLD2, Dates, Plots

using ITransverse

ITensors.enable_debug_checks()


function main()

    zero_state = Vector{ComplexF64}([1,0])
    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])


    JXX = 1.0  
    hz = 0.4
    dt = 0.1

    nbeta=0

    init_state = plus_state

    SVD_cutoff = 1e-10
    maxbondim = 60
    itermax = 400
    verbose=false
    ds2_converged=1e-6

    params = pparams(JXX, hz, dt, nbeta, init_state)
    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []
    evs_x2 = []
    evs_z2 = []


    evs_xs = []
    evs_zs = []
    evs_x2s = []
    evs_z2s = []

    leftvecs = []
    ds2s = []

    ts = 70:10:70

    #TODO 
    renyi2 = []
    renyi2alt = []
    for Nsteps in ts

        time_sites = siteinds("S=3/2", Nsteps)

        #test_mps = productMPS(time_sites,"+")
        #test_mps = productMPS(time_sites,"↑")

        init_mps = build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

        mpo_X = build_ising_folded_tMPO(build_expH_ising_murg, params, sigX, time_sites)
        mpo_Z = build_ising_folded_tMPO(build_expH_ising_murg, params, sigZ, time_sites)
        mpo_1 = build_ising_folded_tMPO(build_expH_ising_murg, params, Id, time_sites)


        ll, rr, lO, Or, ds2_pm, dns  = powermethod(init_mps, mpo_1, mpo_X, pm_params) # kwargs)

        @info "Checking (l|Or)"
        ITransverse.check_gencan_left_sym_phipsi(ll,Or)

        @info "Checking (lO|r)"
        ITransverse.check_gencan_left_sym_phipsi(lO,rr)

        renyi2 = ITransverse.ExtraUtils.generalized_renyi_entropy(ll,Or, 2)
        renyi2alt = ITransverse.ExtraUtils.generalized_renyi_entropy(lO,rr, 2)

        @show renyi2
        @show renyi2alt

        llalt, ds2_pm  = powermethod_Lonly(init_mps, mpo_1, mpo_X, pm_params) 

        @show inner(ll, llalt)
        @show norm(ll)
        @show norm(llalt)
        @show(norm(ll)^2 + norm(llalt)^2 - 2*inner(ll, llalt) )
        #sleep(10)

        ev0 = overlap_noconj(ll, rr)

        lz = apply(mpo_Z, ll, alg="naive", truncate=false)
        lx = apply(mpo_X, ll, alg="naive", truncate=false)
        lid = apply(mpo_1, ll, alg="naive", truncate=false)

        evz = overlap_noconj(lz, rr)
        evx = overlap_noconj(lx, rr)
        ev1 = overlap_noconj(lid, rr)


        evx2 = evx/ev1
        evz2 = evz/ev1

        lzs = apply(mpo_Z, llalt, alg="naive", truncate=false)
        lxs = apply(mpo_X, llalt, alg="naive", truncate=false)
        lids = apply(mpo_1, llalt, alg="naive", truncate=false)

        evzs = overlap_noconj(lzs, llalt)
        evxs = overlap_noconj(lxs, llalt)
        ev1s = overlap_noconj(lids, llalt)

        #@show ev1 

        evx2s = evxs/ev1s
        evz2s = evzs/ev1s

        push!(evs_x, evx)
        push!(evs_z, evz)

        push!(evs_x2, evx2)
        push!(evs_z2, evz2)


        push!(evs_xs, evxs)
        push!(evs_zs, evzs)

        push!(evs_x2s, evx2s)
        push!(evs_z2s, evz2s)


        push!(leftvecs, ll)
        push!(ds2s, ds2_pm)

        
        # if Nsteps%10 == 0
        #     out_filename = "checkpoint_pm_fold_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"
        #     @info "saving cp $out_filename after $Nsteps steps"
        #     jldsave(out_filename; leftvecs, evs_z, evs_x, ds2s, params, pm_params)
        # end


    end

    return leftvecs, 
    evs_z, evs_x, evs_x2, evs_z2, 
    evs_zs, evs_xs, evs_x2s, evs_z2s, 
    renyi2, renyi2alt,
    ds2s, ts 
end



leftvecs, evs_z, evs_x, evs_x2, evs_z2, evs_zs, evs_xs, evs_x2s,
 evs_z2s, renyi2, renyi2alt, ds2s, ts= main()

scatter(ts, real(evs_x),label="<X>", legend=:left)
scatter!(ts, real(evs_z), label="<Z>")

scatter!(ts, real(evs_x2),label="<Xn>", legend=:left)
scatter!(ts, real(evs_z2), label="<Zn>")
#jldsave("out_folded_pm_ising.jld2"; leftvecs, evs_z, evs_x, ds2s)


scatter!(ts, real(evs_xs),label="<Xs>", legend=:left)
scatter!(ts, real(evs_zs), label="<Zs>")

scatter(ts, real(evs_x2s),label="<Xsn>", legend=:left)
scatter!(ts, real(evs_z2s), label="<Zsn>")


plot!(ITransverse.ExtraUtils.bench_X_04_plus[1:80], label=nothing)
plot!(ITransverse.ExtraUtils.bench_Z_04_plus[1:80], label=nothing)