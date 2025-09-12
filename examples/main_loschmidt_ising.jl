using ITensors
using ITensorMPS
using ITransverse
using ITransverse: plus_state, up_state

function ising_loschmidt(tp::tMPOParams, Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    cutoff = 1e-14
    maxbondim = 128
    itermax = 800
    eps_converged = 1e-6

    truncp = TruncParams(cutoff, maxbondim)
    pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM")

    ll_murgs = Vector{MPS}()
    ds2s = [] # Vector{Float64}[]
    leading_eigs = ComplexF64[]
    leading_eigsq = ComplexF64[]
    overlapsLR = ComplexF64[]
    entropies = [] 
    maxents = []

    Ntime_steps = Tstart
    Nsteps = Ntime_steps + 2 * nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(tp.nbeta) imag steps ")
    @info ("Initial state $(tp.bl)")


    b = FwtMPOBlocks(tp)

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps + nbeta

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        mpo = fw_tMPO(b, time_sites, tr=tp.bl)
        start_mps = fw_tMPS(b, time_sites; tr=tp.bl, LR=:right)

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)


        # Entropies 
        sgen = generalized_vn_entropy_symmetric(psi_trunc)
        sgen_sv = generalized_svd_vn_entropy_symmetric(psi_trunc)


        svn = vn_entanglement_entropy(psi_trunc)

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
        #push!(entropies, [sgen, sgen_sv, svn])
        push!(entropies, sgen)
        push!(maxents, [maximum(real(sgen)), maximum(real(sgen_sv)), maximum(svn)])


        curr_T = ts


        # if ts % 40 == 0
        #     out_filename = "cp_ising_$(ts)_$(maxlinkdim(psi_trunc)).jld2"
        #     jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents)
        # end

    end

    return ll_murgs, ds2s, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents


end

function main_ising_loschmidt()

    
    JXX = 1.0
    hz = 1.0
    gx = 0.0
    #H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X


    dt = 0.05

    nbeta = 8

    # init_state = plus_state
    init_state = up_state

    mp = IsingParams(JXX, hz, gx)


    @info ("Initial state $(init_state)  => quench @ $(mp) ")
    
    Tmin = 0
    Tmax = 1
    Tstep = 6


    tp = tMPOParams(dt,  ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state)
    psis1, ds2s, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents = ising_loschmidt(tp, Tmin, Tmax, nbeta; Tstep)

    rr2s = []
    ir2s = []
    r2s= []
    for psi in psis1
        vn = ITransverse.generalized_vn_entropy_symmetric(psi, normalize_eigs=true)
        r2 = ITransverse.generalized_r2_entropy_symmetric(psi, normalize_eigs=true)

        r2 = -log.(r2)

        push!(r2s, r2)
        push!(rr2s, maximum(real(r2)))
        push!(ir2s, maximum(imag(r2)))
    end

    return collect(Tmin:Tstep:Tmax), rr2s, ir2s, entropies, r2s 


end

results = main_ising_loschmidt();