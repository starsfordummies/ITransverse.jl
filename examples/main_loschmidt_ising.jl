using ITensors, JLD2
using ITensorMPS
using ITransverse
using ITransverse: plus_state

function ising_loschmidt(tp::tMPOParams, Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    cutoff = 1e-10
    maxbondim = 60
    itermax = 500
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

        mpo, start_mps = fw_tMPO(b, time_sites, tr=tp.bl)

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
        push!(entropies, [sgen, sgen_sv, svn])
        push!(maxents, [maximum(real(sgen)), maximum(real(sgen_sv)), maximum(svn)])


        curr_T = ts


        if ts % 40 == 0
            out_filename = "cp_ising_$(ts)_$(maxlinkdim(psi_trunc)).jld2"
            jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents)
        end

    end

    return ll_murgs, ds2s, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents


end

function main_ising_loschmidt()

    
    JXX = 1.0
    hz = 1.0
    gx = 0.0
    #H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X


    dt = 0.1

    nbeta = 2

    init_state = plus_state
    #init_state = zero_state

    mp = IsingParams(JXX, hz, gx)


    @info ("Initial state $(init_state)  => quench @ $(mp) ")
    
    Tmin = 20
    Tmax = 38
    Tstep = 2


    tp = tMPOParams(dt,  ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state)
    psis1, ds2s, leading_eigs, leading_eigsq, overlapsLR, entropies, maxents = ising_loschmidt(tp, Tmin, Tmax, nbeta; Tstep)

    rr2s = []
    ir2s = []
    for psi in psis1
        r2 = rtm2_contracted(psi, psi, normalize_factor=overlap_noconj(psi,psi))
        r2 = -log.(r2)
        push!(rr2s, maximum(real(r2)))
        push!(ir2s, maximum(imag(r2)))
    end

    return collect(Tmin:Tstep:Tmax), rr2s, ir2s, entropies 

    # tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_murg_new, mp, nbeta, init_state, init_state)
    # psis2, ds2s, leading_eigs2, leading_eigsq, overlapsLR, entropies2, maxents2 = ising_loschmidt(tp, Tmin, Tmax, nbeta; Tstep)

    # tp = tMPOParams(dt, build_expH_ising_symm_svd, mp, nbeta, init_state, init_state)
    # psis3, ds2s, leading_eigs3, leading_eigsq, overlapsLR, entropies3, maxents3 = ising_loschmidt(tp, Tmin, Tmax, nbeta; Tstep)

    # @show inner(psis1[end],psis2[end])/((norm(psis1[end]))*(norm(psis2[end])))
    # @show inner(psis2[end],psis3[end])/((norm(psis2[end]))*(norm(psis3[end])))
    # @show inner(psis1[end],psis3[end])/((norm(psis1[end]))*(norm(psis3[end])))

    # @show leading_eigs[end], leading_eigs2[end], leading_eigs3[end] 

    #return maxents, maxents2, maxents3, entropies, entropies2, entropies3, leading_eigs

end

results = main_ising_loschmidt();
