using ITensors, JLD2
using ITransverse

function main_ising_loschmidt(Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    JXX = 1.0
    hz = 1.0
    gx = 0.0

    dt = 0.1

    zero_state = Vector{ComplexF64}([1, 0])
    plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])

    init_state = plus_state
    #init_state = zero_state


    cutoff = 1e-12
    maxbondim = 140
    itermax = 500
    eps_converged = 1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTMRDM")

    mp = IsingParams(JXX, hz, gx)
    tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, init_state)

    ll_murgs = Vector{MPS}()
    ds2s = [] # Vector{Float64}[]
    leading_eigs = ComplexF64[]
    leading_eigsq = ComplexF64[]
    overlapsLR = ComplexF64[]

    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(tp.nbeta) imag steps ")
    @info ("Initial state $(tp.bl)")
    @info ("Final state $(tp.tr)")
    @info ("Initial state $(init_state)  => quench @ $(mp) ")


    b = FwtMPOBlocks(tp)

    tpim = tMPOParams(tp; dt=-im*tp.dt)

    b_im = FwtMPOBlocks(tpim)

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps + nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        #mpo, start_mps = fw_tMPO(tp, time_sites)
        #mpo, start_mps = fw_tMPOn(tp, time_sites)
        mpo, start_mps = fw_tMPOn(b, b_im, time_sites)

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)


        leading_eig = inner(conj(psi_trunc'), mpo, psi_trunc)

        # silly extra check so we can see that (LTTR) = lambda^2 (LR)
        OL = apply(mpo, psi_trunc,  alg="naive", truncate=false)
        leading_sq = overlap_noconj(OL, OL)

        normalization = overlap_noconj(psi_trunc,psi_trunc)
        leading_eig, leading_sq, normalization

        push!(ll_murgs, psi_trunc)
        push!(ds2s, ds2)
        push!(leading_eigs, leading_eig)
        push!(leading_eigsq, leading_sq)
        push!(overlapsLR, normalization)

        curr_T = ts


        if ts % 40 == 0
            out_filename = "cp_ising_$(ts)_$(maxlinkdim(psi_trunc)).jld2"
            jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts, leading_eigs, leading_eigsq, overlapsLR)
        end

    end

    return ll_murgs, ds2s, leading_eigs, leading_eigsq, overlapsLR


end


psis, ds2s, leading_eigs, leading_eigsq, overlapsLR = main_ising_loschmidt(10, 160, 2; Tstep=10);
