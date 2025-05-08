using ITensors
using ITransverse
using Test

function test_symmpm(Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    JXX = 1.0
    hz = 1.0
    gx = 0.0

    dt = 0.1

    zero_state = Vector{ComplexF64}([1, 0])
    plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

    #init_state = plus_state
    init_state = zero_state


    cutoff = 1e-14
    maxbondim = 140
    itermax = 800
    eps_converged = 1e-6

    truncp = TruncParams(cutoff, maxbondim)

    mp = IsingParams(JXX, hz, gx)
    tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, init_state)

    b = FwtMPOBlocks(tp)

    tpim = tMPOParams(tp; dt=-im*tp.dt)

    b_im = FwtMPOBlocks(tpim)

    ds2s = Vector{Float64}[]

    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(nbeta) imag steps ")
    @info ("Initial state $(init_state)  => quench @ J=$(JXX) , h=$(hz) ")


    allpsis = Dict(:rdm => MPS[], :eig => MPS[], :svd => MPS[])

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps + nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        #mpo, start_mps = fw_tMPO(tp, time_sites)
        mpo, start_mps = fw_tMPOn(b, b_im, time_sites)

        pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_EIG")

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)
        push!(allpsis[:svd], psi_trunc)

        pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_EIG")

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)

        push!(allpsis[:eig], psi_trunc)

        pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM")
        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)
        push!(allpsis[:rdm], psi_trunc)


        
    end

    return allpsis

end

allpsis = test_symmpm(40, 40, 2)

allents = Dict()

for (k,v) in allpsis
    eigs = diagonalize_rtm_left_gen_sym(v[1]; bring_left_gen=true)
    allents[k] = build_entropies(eigs, [0.5,1,2])
end

@info norm(allents[:eig]["S1.0"])

@test norm(allents[:eig]["S1.0"] - allents[:svd]["S1.0"] )/norm(allents[:eig]["S1.0"]) < 0.001
@test norm(allents[:eig]["S1.0"] - allents[:rdm]["S1.0"] )/norm(allents[:eig]["S1.0"]) < 0.001

@info norm(allents[:eig]["S2.0"])
@test norm(allents[:eig]["S2.0"] - allents[:svd]["S2.0"] )/norm(allents[:eig]["S2.0"]) < 0.001
@test norm(allents[:eig]["S2.0"] - allents[:rdm]["S2.0"] )/norm(allents[:eig]["S2.0"]) < 0.001


@info "Difference EIG-SVD: $(allents[:eig]["S1.0"][20] - allents[:svd]["S1.0"][20])"
@info "Difference EIG-RDM: $(allents[:eig]["S1.0"][20] - allents[:rdm]["S1.0"][20])"