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
    eps_converged = 1e-6

    truncp = trunc_params(cutoff, maxbondim, "EIG")

    pm_params = PMParams(truncp, itermax, eps_converged, true, "SYM")

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params(build_expH_ising_murg, mp, nbeta, init_state, init_state)

    ll_murgs = Vector{MPS}()

    ds2s = Vector{Float64}[]

    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(nbeta) imag steps ")
    @info ("Initial state $(init_state)  => quench @ J=$(JXX) , h=$(hz) ")


    b = FwtMPOBlocks(tp)

    mpim = model_params(tp.mp; dt=-im*tp.mp.dt)
    tpim = tmpo_params(tp; mp=mpim)

    b_im = FwtMPOBlocks(tpim)

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps + nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        #mpo, start_mps = fw_tMPO(tp, time_sites)
        #mpo, start_mps = fw_tMPOn(tp, time_sites)
        mpo, start_mps = fw_tMPOn(b, b_im, time_sites)

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

psis, ds2s = main_ising_loschmidt(50, 50, 2)
