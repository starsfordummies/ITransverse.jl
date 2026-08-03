using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra

using ITransverse: transpose_arrows, arrows_clash, arrow_match, stored_ind

# QN (SzParity) conservation for the unfolded transverse machinery.
# The Z2 conserved here is P = prod(Z): XX and Z commute with it, a longitudinal X field
# does not, so only the hpar=0 line can carry quantum numbers.

const MP = IsingParams(1.0, 0.7, 0.0)
const DT = 0.1
const UP = ComplexF64[1, 0]

""" blocks + tMPO + both edge tMPS, with or without QNs """
function transverse_setup(Nt::Int; qns::Bool, tr=UP)
    ss = siteinds("S=1/2", 3; conserve_szparity=qns)
    U3 = build_Ut(ss, Murg(), MP; dt=DT)
    b = FwtMPOBlocks(U3; init_state=UP)
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:Nt]
    return b, fw_tMPO(b, ts; tr), fw_tMPS(b, ts; LR=:left, tr), fw_tMPS(b, ts; LR=:right, tr)
end

function contract_columns(ll::MPS, mpo::MPO, rr::MPS, L::Int)
    r = rr
    for _ in 1:(L-2)
        r = applyn(mpo, r)
    end
    return overlap_noconj(ll, r)
end


@testset "QN time evolution operator" begin
    ss = siteinds("S=1/2", 4; conserve_szparity=true)
    U = build_Ut(ss, Murg(), MP; dt=DT)

    @test hasqns(U)
    @test all(flux(U[j]) isa QN for j in eachindex(U))

    # same operator as without QNs
    ssp = siteinds("S=1/2", 4)
    Up = build_Ut(ssp, Murg(), MP; dt=DT)
    psi_q = MPS(ComplexF64, ss, "Up")
    psi_p = MPS(ComplexF64, ssp, "Up")
    @test inner(psi_q, apply(U, psi_q)) ≈ inner(psi_p, apply(Up, psi_p))

    # a longitudinal field breaks the parity: clear error, not "Fluxes not all equal"
    @test_throws ErrorException build_Ut(ss, Murg(), IsingParams(1.0, 0.7, 0.3); dt=DT)
    # ... and is fine without QNs
    @test hasqns(build_Ut(ssp, Murg(), IsingParams(1.0, 0.7, 0.3); dt=DT)) == false
end


@testset "QN rotation: arrows of the transverse blocks" begin
    ss = siteinds("S=1/2", 3; conserve_szparity=true)
    U3 = build_Ut(ss, Murg(), MP; dt=DT)
    b = FwtMPOBlocks(U3; init_state=UP)

    @test hasqns(b.Wc)
    # a tMPO column is an MPO: its two site legs, and its two links, must be dual pairs
    @test dir(b.iP) != dir(b.iPs)
    @test dir(b.iL) != dir(b.iR)
    @test noprime(b.iPs) == dag(b.iP)

    # the edge blocks carry their own arrows, hence `stored_ind` in the builders
    @test dir(stored_ind(b.Wr, b.iP)) != dir(stored_ind(b.Wc, b.iP))

    # helpers are exact no-ops without QNs
    i = Index(3, "plain")
    @test arrow_match(i, i) === i
    a = random_itensor(ComplexF64, i, sim(i))
    @test transpose_arrows(a) ≈ a
end


@testset "QN transverse contraction == real-space reference" begin
    L = 5
    Nt = 3
    ss = siteinds("S=1/2", L; conserve_szparity=true)
    U = build_Ut(ss, Murg(), MP; dt=DT)

    psi = MPS(ComplexF64, ss, "Up")
    ev = psi
    for _ in 1:Nt
        ev = apply(U, ev)
    end
    ref = inner(psi, ev)      # <up|U^Nt|up>, the bra convention of the builders

    b, T, LL, RR = transverse_setup(Nt; qns=true)
    @test hasqns(T) && hasqns(LL) && hasqns(RR)
    @test contract_columns(LL, T, RR, L) ≈ ref

    # and identical to the same network built without QNs
    _, Tp, Lp, Rp = transverse_setup(Nt; qns=false)
    @test contract_columns(LL, T, RR, L) ≈ contract_columns(Lp, Tp, Rp, L)
end


@testset "QN boundary states must lie in one symmetry sector" begin
    b, _, _, _ = transverse_setup(3; qns=true)
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:3]

    @test hasqns(fw_tMPO(b, ts; tr=UP))                       # |up>   : one block
    @test hasqns(fw_tMPO(b, ts; tr=ComplexF64[0, 1]))         # |down> : one block
    # |+> spans both parity blocks and has no definite flux
    @test_throws ErrorException fw_tMPO(b, ts; tr=ComplexF64[1, 1] / sqrt(2))
end


@testset "QN power method" begin
    Nt = 6
    L = 5

    results = Dict{Bool,Any}()
    for qns in (false, true)
        b, T, LL, RR = transverse_setup(Nt; qns)
        pmp = PMParams(; truncp=(; cutoff=1e-14, maxdim=64, alg="naive"), itermax=60,
                       eps_converged=1e-11, opt_method=:sym, normalization="norm")
        psi, _ = powermethod_sym(RR, T, pmp)
        lead = overlap_noconj(psi, applyn(T, psi)) / overlap_noconj(psi, psi)
        # gauge-invariant check: the ordinary (conjugating) entanglement spectrum. The
        # *generalized* entropies are only gauge invariant in generalized canonical form,
        # which has no QN implementation, so they are not comparable across the two runs.
        results[qns] = (; psi, lead, vn=vn_entanglement_entropy(psi))
    end

    @test hasqns(results[true].psi)
    @test !hasqns(results[false].psi)
    # the QN and plain power methods must find the same fixed point
    @test isapprox(abs(results[true].lead), abs(results[false].lead); rtol=1e-6)
    @test isapprox(results[true].vn[end÷2], results[false].vn[end÷2]; rtol=1e-4)

    # a state can be contracted with itself again (arrows reversed, data untouched)
    psi = results[true].psi
    @test arrows_clash(psi, psi)
    @test overlap_noconj(psi, psi) ≈ overlap_noconj(transpose_arrows(psi), psi)

    # truncation via the ITensors-native algorithms works with QNs
    for alg in ("naive", "densitymatrix")
        b, T, LL, RR = transverse_setup(Nt; qns=true)
        out, _ = tapply(T, RR; alg, cutoff=1e-14, maxdim=32)
        @test hasqns(out)
        @test overlap_noconj(LL, out) ≈ overlap_noconj(LL, applyn(T, RR))
    end
end


@testset "QN: dense-only routines fail loudly instead of dropping symmetry" begin
    b, T, LL, RR = transverse_setup(4; qns=true)

    @test_throws ErrorException gen_canonical(RR, length(RR))
    @test_throws ErrorException diagonalize_rtm_symmetric(RR; bring_gen_can=true)

    i = Index([QN("SzParity", 0, 2) => 2, QN("SzParity", 1, 2) => 2], "i")
    a = random_itensor(ComplexF64, i, dag(prime(i)))
    @test_throws ErrorException ITransverse.ceigen(a, i)
    @test_throws ErrorException ITransverse.symm_oeig(a, i)
    # symm_svd *is* implemented for QNs, see test_qn_symmetric.jl
    @test ITransverse.symm_svd(a, i) isa ITensors.TruncSVD

    # ... while the QN-capable paths still work on the same state
    @test length(diagonalize_rtm_symmetric(RR; bring_gen_can=false)) == length(RR) - 1
    @test length(vn_entanglement_entropy(RR)) == length(RR) - 1
end
