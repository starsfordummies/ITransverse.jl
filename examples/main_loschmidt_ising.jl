using ITensors
using ITensorMPS
using ITransverse

""" Runs the power method for Loschmidt ising """
function ising_loschmidt(b::FwtMPOBlocks, ts::Int, pm_params)

    tp = b.tp 

    Nsteps     = tp.nbeta + ts
    time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

    mpo       = fw_tMPO(b, time_sites, tr=tp.bl)
    start_mps = fw_tMPS(b, time_sites; tr=tp.bl, LR=:right)

    psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)

    normalization = overlap_noconj(psi_trunc, psi_trunc)
    psi_trunc     = psi_trunc / sqrt(normalization)

    sgen        = gensym_renyi_entropies(psi_trunc)
    leading_eig = inner(conj(psi_trunc'), mpo, psi_trunc)

    # extra check: (LTTR) = lambda^2 (LR)
    OL         = apply(mpo, psi_trunc, alg="naive", truncate=false)
    leading_sq = overlap_noconj(OL, OL)

    return psi_trunc, (; ds2, leading_eig, leading_sq, normalization, entropy=sgen)
end


function main_losch(Ntmin = 10, Ntmax  = 80; Ntstep = 2)

    JXX = 1.0
    hz  = -1.5
    gx  = 0.0

    dt    = 0.1
    dbeta = im*dt   # reversed sign beta imag time

    nbeta      = 4
    init_state = up_state

    mp = IsingParams(JXX, hz, gx)
    @info "Initial state $(init_state) => quench @ $(mp)"

    allts = Ntmin:Ntstep:Ntmax

    tp = tMPOParams(dt, dbeta, Murg(), mp, nbeta, init_state)

    b = FwtMPOBlocks(tp)

    @info "Optimizing for T=$(allts) with $(tp.nbeta) imag steps"

    pm_params = PMParams(;
        cutoffs       = [1e-12],
        maxdims       = 2:2:256,
        itermax       = 2000,
        eps_converged = 1e-9,
        normalization = "overlap",
        stuck_after   = 200,
    )

    psis    = Vector{MPS}()
    results = NamedTuple[]

    for ts in allts
        #@info "  t = $ts"
        psi, res = ising_loschmidt(b, ts, pm_params)
        push!(psis, psi)
        push!(results, res)
    end

    return collect(allts), psis, results
end


times, psis, results = main_losch(10,300; Ntstep=2)