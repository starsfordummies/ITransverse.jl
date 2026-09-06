"""
Boundary-site metadata on transverse boundary vectors.

A *non-product* boundary state is appended as its own site, at the bottom (`rho0` / `bl`)
or at the top (`fold_op` / `tr`). Afterwards nothing in the MPS distinguishes it from a
time site - an appended boundary tensor and an end-of-chain site tensor have the same rank
- so anything that rebuilds a tMPO over "the time sites of this vector" needs the counts
recorded at build time. `SidedMPS` carries them; `sided=true` asks the builders for it.

Before this, `_time_sites` only stripped the bottom, so a non-product *top* leaked into the
rebuilt tMPO: an error when the dimensions clashed, and - with a non-product top on both
vectors - a wrong number with no error at all.
"""

using ITensors, ITensorMPS, ITransverse
using ITransverse: _time_sites, is_product_boundary, n_boundary_sites
using Test

const MP = IsingParams(1.0, 0.95, 1.4)

function folded_setup(Nt::Int)
    tp = tMPOParams(MP; dt=0.1, scheme=Murg(), init_state=[1, 0])
    return FoldtMPOBlocks(tp), siteinds(4, Nt)
end

""" A non-product top boundary: the edge tensor of a boundary MPS of bond dimension `chi`. """
function nonproduct_top(chi::Int, d::Int = 4)
    iphys, ibond = Index(d, "phys"), Index(chi, "bnd")
    return boundary_tensor(random_itensor(ComplexF64, iphys, ibond); phys=iphys, right=ibond)
end

@testset "folded_tMPS accepts sided=true" begin
    b, ts = folded_setup(8)
    for LR in (:left, :right)
        s = folded_tMPS(b, ts; LR, sided=true)
        @test s isa SidedMPS
        @test side(s) == LR
        # same state, just tagged (fresh link Indices each build, so compare shape)
        plain = folded_tMPS(b, ts; LR)
        @test MPS(s) isa MPS
        @test length(MPS(s)) == length(plain)
        @test siteinds(MPS(s)) == siteinds(plain)
        # product boundaries on both ends here
        @test n_boundary_bottom(s) == 0
        @test n_boundary_top(s) == 0
    end
    # the edge helpers forward their kwargs
    @test folded_left_tMPS(b, ts; sided=true) isa SidedMPS
    @test folded_right_tMPS(b, ts; sided=true) isa SidedMPS
    # and the default is still a plain MPS
    @test folded_tMPS(b, ts) isa MPS
end

@testset "boundary sites are counted" begin
    b, ts = folded_setup(8)
    top = nonproduct_top(3)
    @test !is_product_boundary(top)

    plain = folded_tMPS(b, ts; LR=:right, fold_op=top)
    @test length(plain) == length(ts) + 1        # the top boundary got its own site

    s = folded_tMPS(b, ts; LR=:right, fold_op=top, sided=true)
    @test n_boundary_top(s) == 1
    @test n_boundary_bottom(s) == 0
    @test length(s) == length(ts) + 1

    # metadata survives the operations that keep the chain length
    @test n_boundary_top(copy(s)) == 1
    @test n_boundary_top(transpose(s)) == 1
    @test n_boundary_bottom(transpose(s)) == 0
    @test side(transpose(s)) == :left

    # a plain MPS records nothing, and cannot: this is exactly the case that used to be
    # mistaken for a time site
    @test n_boundary_top(plain) == 0
end

@testset "_time_sites drops both ends" begin
    b, ts = folded_setup(8)
    top = nonproduct_top(3)

    s = folded_tMPS(b, ts; LR=:right, fold_op=top, sided=true)
    @test length(_time_sites(s, b)) == length(ts)
    @test _time_sites(s, b) == siteinds(MPS(s))[1:length(ts)]

    # product top: unchanged, with or without the wrapper
    sp = folded_tMPS(b, ts; LR=:right, sided=true)
    @test length(_time_sites(sp, b)) == length(ts)
    @test length(_time_sites(folded_tMPS(b, ts; LR=:right), b)) == length(ts)
end

@testset "expvals refuse a non-product top boundary" begin
    Nt = 8
    b, ts = folded_setup(Nt)
    # Bond dimension equal to the folded physical dimension, on *both* vectors: this is the
    # combination that used to sail through and return a number. It cannot be right - the
    # rebuilt operator column closes its own top with `fold_op`, so it has no room for the
    # boundary as well - so the helpers must now say so.
    top = nonproduct_top(4)
    ll = folded_tMPS(b, ts; LR=:left,  fold_op=top, sided=true)
    rr = folded_tMPS(b, ts; LR=:right, fold_op=top, sided=true)
    @test n_boundary_top(ll) == n_boundary_top(rr) == 1

    idv = ITransverse.itensor_to_vector(ITransverse.vectorized_identity(Index(4)))
    @test_throws ErrorException expval_LR(ll, rr, idv, b)
    @test_throws ErrorException expval_LR(ll, rr, [1, 0, 0, -1], [1, 0, 0, -1], b)
    @test_throws ErrorException ITransverse.expval_LR_apply(ll, rr, idv, b)
    @test_throws ErrorException compute_expvals(ll, rr, ["Z"], b)

    # and the message points at the boundary, not at some index mismatch downstream
    err = try
        expval_LR(ll, rr, idv, b)
    catch e
        sprint(showerror, e)
    end
    @test occursin("boundary site", err)

    # the plain-MPS path has no metadata, so it cannot refuse - it silently contracts a
    # network in which the boundary site is treated as a time step. Pinned as a reminder
    # that `sided=true` is what buys the check.
    @test expval_LR(MPS(ll), MPS(rr), idv, b) isa Number
end

@testset "expvals still work with a product top" begin
    b, ts = folded_setup(8)
    ll = folded_tMPS(b, ts; LR=:left, sided=true)
    rr = folded_tMPS(b, ts; LR=:right, sided=true)

    idv = ITransverse.itensor_to_vector(ITransverse.vectorized_identity(Index(4)))
    # the wrapper must not change any answer when there is no boundary site to skip
    @test expval_LR(ll, rr, idv, b) ≈ expval_LR(MPS(ll), MPS(rr), idv, b)
    @test compute_expvals(ll, rr, ["Z", "X"], b) == compute_expvals(MPS(ll), MPS(rr), ["Z", "X"], b)
end

@testset "fw_tMPS records its boundary sites too" begin
    up = ComplexF64[1, 0]
    ss = siteinds("S=1/2", 3)
    tp = tMPOParams(IsingParams(1.0, 0.7, 0.0); dt=0.1, scheme=Murg(), nbeta=0, init_state=up)
    b = FwtMPOBlocks(tp)
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:6]

    s = fw_tMPS(b, ts; LR=:right, tr=up, sided=true)
    @test s isa SidedMPS
    @test n_boundary_bottom(s) == n_boundary_sites(tp.bl)
    @test n_boundary_top(s) == 0        # `tr` is a product vector here
end
