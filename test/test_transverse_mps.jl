using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra
using Logging

using ITransverse: transpose_arrows

# `TransverseMPS` keeps the `LR` choice attached to a boundary vector, so that a column can only
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
           fw_tMPS(b, ts; LR=:left,  tr=UPS),
           fw_tMPS(b, ts; LR=:right, tr=UPS),
           unsided(fw_tMPS(b, ts; LR=:left,  tr=UPS)),
           unsided(fw_tMPS(b, ts; LR=:right, tr=UPS))
end

@testset "TransverseMPS bookkeeping (qns=$qns)" for qns in (false, true)
    Nt, L = 5, 5
    T, Ls, Rs, Lp, Rp = sided_setup(Nt; qns)

    @test Ls isa TransverseMPS && side(Ls) === :left
    @test Rs isa TransverseMPS && side(Rs) === :right
    @test length(Ls) == Nt && maxlinkdim(Rs) == maxlinkdim(Rp)
    @test MPS(Rs) isa MPS                    # conversion gives the plain MPS back
    @test convert(MPS, Rs) === MPS(Rs)
    @test hasqns(Rs) == qns
    # the builders are return-strict: a transverse boundary vector is always tagged, and
    # `unsided` is the way back to the bare state
    @test Lp isa MPS && Rp isa MPS       # these went through `unsided` in the setup
    @test unsided(Rs) === MPS(Rs)        # `unsided` is total, `MPS(...)` only for the wrapper
    @test unsided(Rp) === Rp

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
    @test_throws ErrorException TransverseMPS(Rp, :sideways)

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

# The routines that take a boundary vector accept `TMPSorMPS`: a tagged vector goes in
# without an explicit `MPS(...)` at the call site, and must give exactly the same answer as
# the plain one it wraps. The tag is *not* claimed to survive - these return plain `MPS`.
@testset "TMPSorMPS widening agrees with the plain MPS (qns=$qns)" for qns in (false, true)
    Nt = 5
    T, Ls, Rs, Lp, Rp = sided_setup(Nt; qns)

    # scalars / vectors: tagged and plain must agree exactly
    @test overlap_noconj(Ls, Rp)          == overlap_noconj(Lp, Rp)
    @test overlap_noconj(Lp, Rs)          == overlap_noconj(Lp, Rp)
    @test fidelity(Rs, Rp)                == fidelity(Rp, Rp)
    @test logfidelity(Rs, Rs)             == logfidelity(Rp, Rp)
    @test gen_fidelity(Rs, Rp)            == gen_fidelity(Rp, Rp)
    @test vn_entanglement_entropy(Rs)     == vn_entanglement_entropy(Rp)
    @test expval_LR(Ls, T, Rp)            == expval_LR(Lp, T, Rp)
    @test expval_LR(Lp, T, Rs)            == expval_LR(Lp, T, Rp)

    # Return-strict: a sweep leaves a vector on the side it was on, and hands back a tagged
    # one. A plain `MPS` in still gives a plain `MPS` out - the side is never invented.
    swept, sv = truncate_sweep_sym(Rs; cutoff=1e-12, maxdim=8)
    swept_p, sv_p = truncate_sweep_sym(Rp; cutoff=1e-12, maxdim=8)
    @test swept isa TransverseMPS && side(swept) === :right
    @test swept_p isa MPS && sv == sv_p
    @test unsided(swept) ≈ swept_p

    # `tlrapply` returns a left and a right *by contract*, so `TruncLR` carries both tags
    lr = tlrapply(Ls, T, T, Rs; alg="naiveRTM", cutoff=1e-12, maxdim=8)
    lr_p = tlrapply(Lp, T, T, Rp; alg="naiveRTM", cutoff=1e-12, maxdim=8)
    @test lr.L isa TransverseMPS && side(lr.L) === :left
    @test lr.R isa TransverseMPS && side(lr.R) === :right
    @test lr_p.L isa MPS && lr_p.R isa MPS
    @test lr.sv == lr_p.sv
    # and the pair it returns can be contracted without any unwrapping
    @test overlap_noconj(lr.L, lr.R) ≈ overlap_noconj(lr_p.L, lr_p.R)

    # `gen_canonical` (and the entropies routed through it or through `renyi_entropies`)
    # does not support QNs - for a plain `MPS` either, so this is not about the widening
    if !qns
        @test renyi_entropies(Rs) == renyi_entropies(Rp)
        @test gen_canonical(Rs, 2) isa TransverseMPS
        @test unsided(gen_canonical(Rs, 2)) ≈ gen_canonical(Rp, 2)
        @test gensym_renyi_entropies(Rs) == gensym_renyi_entropies(Rp)
        @test diagonalize_rtm_symmetric(Rs) == diagonalize_rtm_symmetric(Rp)
    end

    # `applyn` / `applyns` name the legs they contract, so they stay the plain-`MPS`
    # primitives: on a tagged vector they refuse and point at `apply_column`, rather than
    # silently substituting the other one.
    @test_throws ErrorException applyn(T, Rs)
    @test_throws ErrorException applyns(T, Ls)
    for f in (applyn, applyns)
        err = try f(T, Rs) catch e; sprint(showerror, e) end
        @test occursin("apply_column", err)
    end

    # an MPO is not a boundary vector, and TMPSorMPS must not admit one
    @test !(T isa ITransverse.TMPSorMPS)
end
