using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra
using Logging

using ITransverse: transpose_arrows

# `SidedMPS` keeps the `LR` choice attached to a boundary vector, so that a column can only
# be applied from the correct side and two same-side vectors cannot be paired.

const MPS_ = IsingParams(1.0, 0.7, 0.0)
const UPS = ComplexF64[1, 0]

function sided_setup(Nt::Int; qns::Bool)
    ss = siteinds("S=1/2", 3; conserve_szparity=qns)
    U3 = with_logger(NullLogger()) do
        build_Ut(ss, Murg(), MPS_; dt=0.1)
    end
    b = with_logger(NullLogger()) do
        FwtMPOBlocks(U3; init_state=UPS)
    end
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:Nt]
    T = fw_tMPO(b, ts; tr=UPS)
    return T,
           fw_tMPS(b, ts; LR=:left,  tr=UPS, sided=true),
           fw_tMPS(b, ts; LR=:right, tr=UPS, sided=true),
           fw_tMPS(b, ts; LR=:left,  tr=UPS),
           fw_tMPS(b, ts; LR=:right, tr=UPS)
end

@testset "SidedMPS bookkeeping (qns=$qns)" for qns in (false, true)
    Nt, L = 5, 5
    T, Ls, Rs, Lp, Rp = sided_setup(Nt; qns)

    @test Ls isa SidedMPS && side(Ls) === :left
    @test Rs isa SidedMPS && side(Rs) === :right
    @test length(Ls) == Nt && maxlinkdim(Rs) == maxlinkdim(Rp)
    @test MPS(Rs) isa MPS                    # conversion gives the plain MPS back
    @test convert(MPS, Rs) === MPS(Rs)
    @test hasqns(Rs) == qns
    # the plain builders are unaffected by the opt-in
    @test Lp isa MPS && Rp isa MPS

    # apply_column picks applyn / applyns from the side. (Compared through overlaps: each
    # `applyn` call sim's the link indices, so the tensors themselves are not comparable.)
    @test overlap_noconj(Ls, apply_column(T, Rs)) ≈ overlap_noconj(Lp, applyn(T, Rp))
    @test overlap_noconj(apply_column(T, Ls), Rs) ≈ overlap_noconj(applyns(T, Lp), Rp)
    @test side(apply_column(T, Rs)) === :right

    # a full column contraction matches the plain route exactly
    r, rp = Rs, Rp
    for _ in 1:(L-2)
        r = apply_column(T, r)
        rp = applyn(T, rp)
    end
    @test overlap_noconj(Ls, r) ≈ overlap_noconj(Lp, rp)
    # ... and the sides may be given in either order
    @test overlap_noconj(Ls, r) ≈ overlap_noconj(r, Ls)

    # pairing two vectors from the same side is an implicit transpose: refused
    @test_throws ErrorException overlap_noconj(Rs, Rs)
    @test_throws ErrorException overlap_noconj(Ls, Ls)
    @test_throws ErrorException expval_LR(Rs, T, Rs)
    @test_throws ErrorException SidedMPS(Rp, :sideways)

    # transpose flips the side, and with QNs also the arrows (data untouched)
    tR = transpose(Rs)
    @test side(tR) === :left
    @test norm(MPS(tR)) ≈ norm(MPS(Rs))
    if qns
        @test dir(siteinds(tR)[1]) != dir(siteinds(Rs)[1])
    else
        @test MPS(tR)[1] ≈ MPS(Rs)[1]     # exact no-op without QNs
    end
    # the transposed right vector may legally be paired with the original
    @test overlap_noconj(tR, Rs) ≈ overlap_noconj(MPS(tR), MPS(Rs))

    # truncating application keeps the side
    out, _ = tapply_column(T, Rs; alg="naive", cutoff=1e-14, maxdim=32)
    @test side(out) === :right
    @test overlap_noconj(Ls, out) ≈ overlap_noconj(Lp, applyn(T, Rp))
end
