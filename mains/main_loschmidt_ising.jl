using ITensors, JLD2
using ITransverse

function main_build_ising_loschmidt_ents(Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    JXX = 1.0
    hz = 1.0
    gx = 0.0

    dt = 0.1

    zero_state = Vector{ComplexF64}([1, 0])
    plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

    #init_state = plus_state
    init_state = zero_state


    cutoff = 1e-16
    maxbondim = 140
    itermax = 800
    verbose = false
    ds2_converged = 1e-6


    pm_params = ppm_params(;itermax, cutoff, maxbondim, verbose, ds2_converged, ortho_method="EIG")

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_parallel_field_murg, mp, nbeta, init_state)


    ll_murgs = Vector{MPS}()

    ds2s = Vector{Float64}[]

    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(nbeta) imag steps ")
    @info ("Initial state $(init_state)  => quench @ J=$(JXX) , h=$(hz) ")

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps + nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        #mpo_L, start_mps = build_ising_fw_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)
        #ll_murg_s, ds2s_murg_s = powermethod_sym(start_mps, mpo_L, pm_params)

        mpo_L, start_mps = build_fw_tMPO_regul_beta(tp, time_sites)

        ll_murg_s, ds2s_murg_s = powermethod_sym(start_mps, mpo_L, pm_params)

        # @show inner(ll_murg_s, ll_murg_s2)/norm(ll_murg_s)^2

        push!(ll_murgs, ll_murg_s)
        push!(ds2s, ds2s_murg_s)

        curr_T = ts


        if ts % 20 == 0
            out_filename = "cp_ising_$(ts)_$(maxlinkdim(ll_murg_s)).jld2"
            jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts)
        end

    end

    return ll_murgs, ds2s


end

main_build_ising_loschmidt_ents(40, 70, 2)