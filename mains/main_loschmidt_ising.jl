using ITensors, JLD2
using Revise
using ITransverse

function main_ising_loschmidt(Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

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
    ds2_converged = 1e-6

    truncp = trunc_params(cutoff, maxbondim, "EIG")

    pm_params = PMParams(truncp, itermax, ds2_converged, true)

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params(build_expH_ising_murg, mp, nbeta, init_state, init_state)

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
        #psi_trunc, ds2s_murg_s = powermethod_sym(start_mps, mpo_L, pm_params)

        mpo, start_mps = fw_tMPO(tp, time_sites)

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)

        # @show inner(psi_trunc, psi_trunc2)/norm(psi_trunc)^2

        push!(ll_murgs, psi_trunc)
        push!(ds2s, ds2)

        curr_T = ts


        if ts % 20 == 0
            out_filename = "cp_ising_$(ts)_$(maxlinkdim(psi_trunc)).jld2"
            jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts)
        end

    end

    return ll_murgs, ds2s


end

psis, ds2s = main_ising_loschmidt(40, 40, 2)