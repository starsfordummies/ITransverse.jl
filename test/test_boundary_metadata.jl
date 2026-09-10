"""
Non-product boundary states on transverse boundary vectors.

A *non-product* boundary state is appended as its own site, at the bottom (`rho0` / `bl`)
or at the top (`fold_op` / `tr`), and afterwards has the same rank as an end-of-chain site
tensor. Nothing records the counts: the one place that needs the bottom one reads it off
the blocks it is handed (`n_boundary_sites(b.rho0)`), and on that path a vector's top is
always the product `vectorized_identity`, since operators are inserted by the expval
helpers rather than built into the vector.

A vector that *does* carry a non-product top belongs to a hand-contracted network. The
helpers refuse it on a dimension check - which catches every bond dimension except one
equal to the folded physical dimension.
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

@testset "folded_tMPS is return-strict" begin
    b, ts = folded_setup(8)
    for LR in (:left, :right)
        s = folded_tMPS(b, ts; LR)
        @test s isa TransverseMPS
        @test side(s) == LR
        # `unsided` gives the bare state back (fresh link Indices each build, so compare shape)
        plain = unsided(folded_tMPS(b, ts; LR))
        @test MPS(s) isa MPS
        @test length(MPS(s)) == length(plain)
        @test siteinds(MPS(s)) == siteinds(plain)
        # product boundaries on both ends here, so no extra sites
        @test length(s) == length(ts)
    end
    # the edge helpers forward their kwargs
    @test folded_left_tMPS(b, ts) isa TransverseMPS
    @test folded_right_tMPS(b, ts) isa TransverseMPS
    # there is no untagged spelling any more: a transverse boundary vector always knows
    # its side, and a plain `MPS` means "not a transverse boundary vector"
    @test folded_tMPS(b, ts) isa TransverseMPS
end

@testset "a non-product boundary gets its own site" begin
    b, ts = folded_setup(8)
    top = nonproduct_top(3)
    @test !is_product_boundary(top)

    s = folded_tMPS(b, ts; LR=:right, fold_op=top)
    @test length(s) == length(ts) + 1        # the top boundary got its own site
    @test length(folded_tMPS(b, ts; LR=:right)) == length(ts)   # a product one does not

    # the side is the only metadata, and it survives the length-preserving operations
    @test side(copy(s)) == :right
    @test side(transpose(s)) == :left
end

@testset "_time_sites drops the bottom, read off the blocks" begin
    b, ts = folded_setup(8)

    # product bottom (this `b`): nothing to drop, all sites are time sites
    sp = folded_tMPS(b, ts; LR=:right)
    @test n_boundary_sites(b.rho0) == 0
    @test _time_sites(sp, b) == siteinds(MPS(sp)) == ts
    @test _time_sites(unsided(sp), b) == ts        # tagged or not, same answer

    # a non-product *top* is deliberately NOT trimmed: on this path a vector's top is always
    # the product identity, so the extra site shows up as an extra "time site" and the
    # dimension guard is what rejects it (see the testset below)
    s = folded_tMPS(b, ts; LR=:right, fold_op=nonproduct_top(3))
    @test length(_time_sites(s, b)) == length(ts) + 1
    @test dim(last(_time_sites(s, b))) == 3 != dim(b.iP)
end

@testset "expvals refuse a non-product top boundary (when they can see it)" begin
    Nt = 8
    b, ts = folded_setup(Nt)
    idv = ITransverse.itensor_to_vector(ITransverse.vectorized_identity(Index(4)))
    d = dim(b.iP)

    # A boundary bond of a different dimension than the folded physical one is visible as a
    # site that cannot be a time step, so the helpers refuse.
    for chi in (3, 7)
        @test chi != d
        ll = folded_tMPS(b, ts; LR=:left,  fold_op=nonproduct_top(chi))
        rr = folded_tMPS(b, ts; LR=:right, fold_op=nonproduct_top(chi))
        @test length(rr) == length(ts) + 1
        @test_throws ErrorException expval_LR(ll, rr, idv, b)
        @test_throws ErrorException expval_LR(ll, rr, [1, 0, 0, -1], [1, 0, 0, -1], b)
        @test_throws ErrorException compute_expvals(ll, rr, ["Z"], b)
        err = try expval_LR(ll, rr, idv, b) catch e; sprint(showerror, e) end
        # the message names the boundary, not some index mismatch downstream
        @test occursin("non-product boundary state", err)
    end

    # A boundary bond equal to the folded physical dimension is invisible to the dimension
    # check, so it gets past the guard - but it does not produce a plausible number either:
    # the two boundary legs stay dangling and the contraction fails in `scalar`. The failure
    # is therefore loud but badly explained, which is the residual limitation.
    ll4 = folded_tMPS(b, ts; LR=:left,  fold_op=nonproduct_top(d))
    rr4 = folded_tMPS(b, ts; LR=:right, fold_op=nonproduct_top(d))
    err4 = try (expval_LR(ll4, rr4, idv, b); "NO ERROR") catch e; sprint(showerror, e) end
    @test err4 != "NO ERROR"
    @test occursin("not a scalar", err4)   # a DimensionMismatch, not the guard
end

@testset "expvals still work with a product top" begin
    b, ts = folded_setup(8)
    ll = folded_tMPS(b, ts; LR=:left)
    rr = folded_tMPS(b, ts; LR=:right)

    idv = ITransverse.itensor_to_vector(ITransverse.vectorized_identity(Index(4)))
    # the wrapper must not change any answer when there is no boundary site to skip
    @test expval_LR(ll, rr, idv, b) ≈ expval_LR(MPS(ll), MPS(rr), idv, b)
    @test compute_expvals(ll, rr, ["Z", "X"], b) == compute_expvals(MPS(ll), MPS(rr), ["Z", "X"], b)
end

@testset "fw_tMPS tags its side too" begin
    up = ComplexF64[1, 0]
    ss = siteinds("S=1/2", 3)
    tp = tMPOParams(IsingParams(1.0, 0.7, 0.0); dt=0.1, scheme=Murg(), nbeta=0, init_state=up)
    b = FwtMPOBlocks(tp)
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:6]

    s = fw_tMPS(b, ts; LR=:right, tr=up)
    @test s isa TransverseMPS && side(s) == :right
    # `bl` and `tr` are both product vectors here, so no extra sites
    @test is_product_boundary(tp.bl) && length(s) == length(ts)
end
