using ITensors
using ITensorMPS
using ITransverse
using ITransverse: plus_state, up_state

""" Builds the dominant vector for unfolded Ising using power method"""
function ising_loschmidt(tp::tMPOParams, Tstart::Int, Tend::Int, nbeta::Int; Tstep::Int=1)

    alg = "RTMsym"
    cutoffs = [1e-12]
    maxdims = 2:2:128
    itermax = 3000
    eps_converged = 1e-9

    truncp = (;cutoff, maxdim, alg)
    #pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM", "overlap", true, stuck_after)
    pm_params = PMParams(;truncp, cutoffs, maxdims, itermax, eps_converged, normalization="overlap", stuck_after=200)


    ll_murgs = Vector{MPS}()
    ds2s = [] # Vector{Float64}[]
    leading_eigs = ComplexF64[]
    leading_eigsq = ComplexF64[]
    overlapsLR = ComplexF64[]
    entropies = [] 
    maxents = []

    # Ntime_steps = Tstart
    # Nsteps = Ntime_steps + nbeta

    allts = Tstart:Tstep:Tend

    @info ("Optimizing for T=$(allts) with $(tp.nbeta) imag steps ")
    @info ("Initial state $(tp.bl)")


    b = FwtMPOBlocks(tp)

    for ts in allts

        Ntime_steps = ts
        Nsteps = nbeta + Ntime_steps 

        time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

        mpo = fw_tMPO(b, time_sites, tr=tp.bl)
        start_mps = fw_tMPS(b, time_sites; tr=tp.bl, LR=:right)

        psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)

        normalization = overlap_noconj(psi_trunc,psi_trunc)
        psi_trunc = psi_trunc/sqrt(normalization)

        # Entropies 
        sgen = gensym_renyi_entropies(psi_trunc)
        #svn = vn_entanglement_entropy(psi_trunc)

        leading_eig = inner(conj(psi_trunc'), mpo, psi_trunc)

        # extra check so we can see that (LTTR) = lambda^2 (LR)
        OL = apply(mpo, psi_trunc,  alg="naive", truncate=false)
        leading_sq = overlap_noconj(OL, OL)

        push!(ll_murgs, psi_trunc)
        push!(ds2s, ds2)
        push!(leading_eigs, leading_eig)
        push!(leading_eigsq, leading_sq)
        push!(overlapsLR, normalization)
        push!(entropies, sgen)


    end

    return ll_murgs, (;ds2s, leading_eigs, leading_eigsq, overlapsLR, entropies)


end



function main_losch()

    
    JXX = 1.0
    hz = 1.0
    gx = 0.0
    #H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X

    dt = 0.1
    dbeta = -im*dt # normal imag time 
    dbeta = im*dt # reversed sign beta imag time 

    nbeta = 4

    # init_state = plus_state
    init_state = up_state

    mp = IsingParams(JXX, hz, gx)

    @info ("Initial state $(init_state)  => quench @ $(mp) ")
    
    Tmin = 60
    Tmax = 60
    Tstep = 1


    tp = tMPOParams(dt, dbeta, Murg(), mp, nbeta, init_state)
    psis, results = ising_loschmidt(tp, Tmin, Tmax, nbeta; Tstep)

    rr2s = []
    ir2s = []
    r2s= []
    # for psi in psis1
    #     vn = ITransverse.generalized_vn_entropy_symmetric(psi, normalize_eigs=true)
    #     r2 = ITransverse.generalized_r2_entropy_symmetric(psi, normalize_eigs=true)

    #     r2 = -log.(r2)

    #     push!(r2s, r2)
    #     push!(rr2s, maximum(real(r2)))
    #     push!(ir2s, maximum(imag(r2)))
    # end

    return collect(Tmin:Tstep:Tmax), psis, results


end


times, psis, results = main_losch()