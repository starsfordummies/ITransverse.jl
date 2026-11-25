using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra
using Logging

using ITransverse: expH_potts_murg, expH_potts_symmetric_svd

# Z3 QN conservation for the 3-state Potts chain. With QNs the model must be written in the
# form where the Z3 generator is diagonal (coupling=:tau, `S1_Z3` sites), and the gates must
# be Murg's: the symmetric-SVD bond basis mixes the charges ±1. See test_qn_ising.jl for the
# Z2 counterpart.

const MP_TAU = PottsParams(1.0, 0.7; coupling=:tau)
const DT = 0.1
const NEUTRAL = ComplexF64[0, 0, 1]   # Σ = 1 eigenstate: Z3 neutral, C-invariant

quiet(f) = with_logger(f, NullLogger())

function potts_setup(Nt::Int; qns::Bool, init=NEUTRAL, nbeta=0)
    ps = siteind("S1_Z3"; conserve_qns=qns)
    mp = PottsParams(MP_TAU.JSS, MP_TAU.ftau; coupling=:tau, phys_site=ps)
    b = quiet(() -> FwtMPOBlocks(tMPOParams(mp; dt=DT, nbeta, scheme=Murg(), init_state=init)))
    ts = [sim(b.iP; tags="Site,time,t=$i") for i in 1:Nt+nbeta]
    return b, fw_tMPO(b, ts; tr=b.tp.bl), fw_tMPS(b, ts; LR=:left, tr=b.tp.bl),
           fw_tMPS(b, ts; LR=:right, tr=b.tp.bl)
end

function contract_columns(ll, mpo, rr, L)
    r = rr
    for _ in 1:(L-2)
        r = apply_column(mpo, r)
    end
    return overlap_noconj(ll, r)
end


@testset "Potts gates: Murg == SymSVD, both forms, and the QN Murg" begin
    for coupling in (:sigma, :tau)
        s = siteinds("S1_Z3", 4)
        mp = PottsParams(1.0, 0.7; coupling)
        for dt in (DT, -im * DT)
            @test norm(prod(expH_potts_murg(s, mp; dt)) - prod(expH_potts_symmetric_svd(s, mp; dt))) < 1e-12
        end
        # 2nd-order Trotter: local error ∝ dt³
        Hd = prod(H_potts(s, mp))
        err(dt) = norm(prod(expH_potts_murg(s, mp; dt)) - exp(-im * dt * Hd))
        @test 7 < err(0.1) / err(0.05) < 9
    end

    # :sigma and :tau are the same model
    function spectrum(mp)
        s = siteinds("S1_Z3", 4)
        co, ci = combiner(prime.(s)...), combiner(s...)
        eigvals(Hermitian(matrix(prod(H_potts(s, mp)) * co * ci, combinedind(co), combinedind(ci))))
    end
    @test spectrum(PottsParams(1.0, 0.7)) ≈ spectrum(MP_TAU)

    sq = siteinds("S1_Z3", 4; conserve_qns=true)
    for dt in (DT, -im * DT)
        Uq = expH_potts_murg(sq, MP_TAU; dt)
        @test hasqns(Uq)
        @test all(flux(Uq[j]) == QN("Z", 0, 3) for j in 1:4)
        @test norm(dense(prod(Uq)) - prod(expH_potts_murg(removeqns(sq), MP_TAU; dt))) < 1e-13
    end

    # the Σ-diagonal form has no Z3 QNs, nor do symmetric-SVD gates: clear errors
    @test_throws ErrorException expH_potts_murg(sq, PottsParams(1.0, 0.7); dt=DT)
    @test_throws ErrorException expH_potts_symmetric_svd(sq, MP_TAU; dt=DT)
end


@testset "Potts column: symmetric for a neutral boundary, Tᵀ = C T C otherwise" begin
    C = [1 0 0; 0 0 1; 0 1 0]       # charge conjugation p → -p
    for (init, symmetric) in ((NEUTRAL, true), (ComplexF64[1, 0, 0], false))
        _, T, _, _ = potts_setup(3; qns=false, init)
        Td = prod(T)
        ts = [only(siteinds(T, j; plev=0)) for j in 1:length(T)]
        Tt = swapprime(Td, 0 => 1; tags="time")
        CTC = Td
        for t in ts
            CTC = replaceinds(CTC * ITensor(C, t, prime(t, 3)) * ITensor(C, t', prime(t, 4)),
                              (prime(t, 3), prime(t, 4)), (t, t'))
        end
        @test norm(CTC - Tt) / norm(Td) < 1e-12
        @test (norm(Td - Tt) / norm(Td) < 1e-12) == symmetric
    end
end


@testset "QN Potts transverse contraction == real-space reference" begin
    L, Nt = 5, 3
    ss = siteinds("S1_Z3", L; conserve_qns=true)
    U = expH_potts_murg(ss, MP_TAU; dt=DT)
    psi = MPS(ComplexF64, ss, fill("Dn", L))    # "Dn" = [0,0,1]
    ev = psi
    for _ in 1:Nt
        ev = apply(U, ev)
    end
    ref = inner(psi, ev)

    _, T, LL, RR = potts_setup(Nt; qns=true)
    @test hasqns(T) && hasqns(LL) && hasqns(RR)
    @test contract_columns(LL, T, RR, L) ≈ ref
    _, Tp, Lp, Rp = potts_setup(Nt; qns=false)
    @test contract_columns(Lp, Tp, Rp, L) ≈ ref
end
