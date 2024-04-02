include("../initialize.jl")

using Revise

using LinearAlgebra, ITensors, JLD2, Dates, Plots


# includet("../itransverse.jl")
# using .ITransverse

using ITransverse

ITensors.enable_debug_checks()


function main()

    zero_state = Vector{ComplexF64}([1,0])
    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])


    JXX = 1.0  
    hz = 0.5
    dt = 0.1
    nbeta=0

    init_state = plus_state

    SVD_cutoff = 1e-8
    maxbondim = 120
    itermax = 400
    verbose=false
    ds2_converged=1e-5

    params = pparams(JXX, hz, dt, nbeta, init_state)
    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []
    evs_xx = []
    leftvecs = []
    ds2s = []

    for Nsteps=4:1:5

        time_sites = siteinds("S=3/2", Nsteps)

        #test_mps = productMPS(time_sites,"+")
        #test_mps = productMPS(time_sites,"↑")

        init_mps = build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

        mpo_X = build_ising_folded_tMPO(build_expH_ising_murg, params, sigX, time_sites)
        mpo_Z = build_ising_folded_tMPO(build_expH_ising_murg, params, sigZ, time_sites)
        mpo_1 = build_ising_folded_tMPO(build_expH_ising_murg, params, Id, time_sites)

        ll, rr, ds2_pm  = powermethod_fold(init_mps, mpo_1, mpo_X, pm_params) # kwargs)

        ev0 = overlap_noconj(ll, rr)

        lz = apply(mpo_Z, ll, alg="naive", truncate=false)
        lx = apply(mpo_X, ll, alg="naive", truncate=false)
        lid = apply(mpo_1, ll, alg="naive", truncate=false)

        evz = overlap_noconj(lz, rr)
        evx = overlap_noconj(lx, rr)

        ev1 = overlap_noconj(lid, rr)

        evx2 = evx/ev1
        evz2 = evz/ev1

        push!(evs_x, evx2)
        push!(evs_z, evz2)
        push!(leftvecs, ll)
        push!(ds2s, ds2_pm)

        
        if Nsteps%10 == 0
            out_filename = "checkpoint_pm_fold_ising_" * Dates.format(now(), "yymmdd_HHMM") * ".jld2"
            @info "saving cp $out_filename after $Nsteps steps"
            jldsave(out_filename; leftvecs, evs_z, evs_x, ds2s, params, pm_params)
        end


    end

    return leftvecs, evs_z, evs_x, ds2s
end



leftvecs, evs_z, evs_x, ds2s = main()

jldsave("out_folded_pm_ising.jld2"; leftvecs, evs_z, evs_x, ds2s)