using ITensors, JLD2
using ITransverse

function main_build_ising_loschmidt_ents(Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    JXX = 1.0
    hz = 1.0

    dt = 0.1

    zero_state = Vector{ComplexF64}([1, 0])
    plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

    init_state = plus_state
    #init_state = zero_state


    SVD_cutoff = 1e-24
    maxbondim = 140
    itermax = 800
    verbose = false
    ds2_converged = 1e-6



    params = pparams(JXX, hz, dt, nbeta, init_state)
    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    out_filename = "out_ents_ising_plus_hp_b$nbeta" * ".jld2"


    ll_murgs = Vector{MPS}()

    ds2s = Vector{Float64}[]


    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta
    time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

    allts = Tstart:Tstep:Tend

    for ts = Tstart:Tstep:Tend

        Ntime_steps = ts
        Nsteps = Ntime_steps + 2 * nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")


        println("Optimizing for $ts timesteps + $nbeta imag steps ")
        println("Initial state $init_state  => quench @ J=$JXX , h=$hz ")

        mpo_L, start_mps = build_ising_fw_tMPO_regul_beta(build_expH_ising_murg, JXX, hz, dt, nbeta, time_sites, init_state)
        
        ll_murg_s, ds2s_murg_s = powermethod_sym(start_mps, mpo_L, pm_params)

        push!(ll_murgs, ll_murg_s)
        push!(ds2s, ds2s_murg_s)

        curr_T = ts


        if ts % 20 == 0
            jldsave(out_filename; nbeta, dt, ll_murgs, ds2s, params, pm_params, curr_T, allts)
        end

    end

end

main_build_ising_loschmidt_ents(30, 50, 2)


