"""
Multi-column expectation values.

`compute_expvals` reaches two-column observables (`XX`, `ZZ`, `eps_ising`) through
`expval_LR(ll, rr, (opL, opR), b)` and `expval_LR_ops`. Neither had a test, and the tuple
entry point forwarded a `match_inds` kwarg that the two-vector method did not accept, so
every two-column observable threw a MethodError.

The Ising epsilon brick is a linear combination of two-site *product* operators, which gives
an independent reference: the spliced-operator path (`expval_LR_ops`) and the product path
must agree.
"""

using ITensors, ITensorMPS, ITransverse
using Test, Logging

const MPε = IsingParams(1.0, 0.95, 1.4)
const Iv = ComplexF64[1, 0, 0, 1]
const Xv = ComplexF64[0, 1, 1, 0]
const Zv = ComplexF64[1, 0, 0, -1]

function folded_pair(Nt::Int; maxdim=64)
    tp = tMPOParams(MPε; dt=0.1, scheme=Murg(), nbeta=0, init_state=[1, 0])
    b = FoldtMPOBlocks(tp)
    ts = siteinds(4, Nt)
    pmp = PMParams(; truncp=(; cutoff=1e-14, maxdim, alg="naive"), itermax=40,
                   eps_converged=1e-11, opt_method=:sym, normalization="norm")
    ll, rr, _ = with_logger(NullLogger()) do
        powermethod_op(folded_right_tMPS(b, ts);
                       mpo_id=folded_tMPO(b, ts), mpo_op=folded_tMPO(b, ts; fold_op=vZ),
                       pm_params=pmp)
    end
    return b, ts, ll, rr
end

@testset "two-column observables are reachable at all" begin
    b, ts, ll, rr = folded_pair(8)
    # the tuple form must agree with the splatted one (this is what used to throw)
    @test expval_LR(ll, rr, (Iv, Iv), b) ≈ expval_LR(ll, rr, Iv, Iv, b)
    for op in ("XX", "ZZ", "eps_ising")
        ev = with_logger(NullLogger()) do; compute_expvals(ll, rr, [op], b) end
        @test haskey(ev, op) && isfinite(abs(ev[op]))
    end
end

@testset "expval_LR_ops agrees with the sum of product two-column expvals" begin
    b, ts, ll, rr = folded_pair(12)
    nrm = expval_LR(ll, rr, (Iv, Iv), b)
    two(a, c) = expval_LR(ll, rr, a, c, b) / nrm

    # eps = Jtwo XX + gperp/2 (IZ + ZI) + hpar/2 (IX + XI)
    ref = MPε.Jtwo * two(Xv, Xv) +
          MPε.gperp / 2 * (two(Iv, Zv) + two(Zv, Iv)) +
          MPε.hpar / 2 * (two(Iv, Xv) + two(Xv, Iv))

    got = with_logger(NullLogger()) do
        compute_expvals(ll, rr, ["eps_ising"], b)["eps_ising"]
    end
    @test isapprox(ref, got; rtol=1e-8)

    # the two columns are not interchangeable, but a symmetric fixed point makes ZI == IZ
    @test two(Zv, Iv) ≈ two(Iv, Zv)
end

@testset "operator over k columns, given as a folded MPS" begin
    b, ts, ll, rr = folded_pair(8)

    # k columns with product tops, staggered prime levels: an independent reference built
    # only from the well-exercised single-column machinery
    function expval_cols(ll, cols::Vector{MPO}, rr)
        k = length(cols); L, R = unsided(ll), unsided(rr)
        O = ITensors.OneITensor()
        for ii in eachindex(L)
            O = O * R[ii]
            for j in k:-1:1; O = O * prime(cols[j][ii], k - j); end
            O = O * prime(L[ii], k)
        end
        return scalar(O)
    end
    col(v) = folded_tMPO(b, ITransverse._time_sites(rr, b); fold_op=v)

    # --- k = 3, bond dimension 3: X1X2 + Z2Z3 ---
    nrm3 = expval_cols(ll, [col(Iv), col(Iv), col(Iv)], rr)
    ref3 = (expval_cols(ll, [col(Xv), col(Xv), col(Iv)], rr) +
            expval_cols(ll, [col(Iv), col(Zv), col(Zv)], rr)) / nrm3

    s3 = siteinds("S=1/2", 3)
    os = OpSum(); os += ("X", 1, "X", 2); os += ("Z", 2, "Z", 3)
    cmb = [combiner(s3[j], s3[j]') for j in 1:3]
    O3 = MPO(os, s3)
    ops3 = MPS([settags(O3[j] * cmb[j], "Site", combinedind(cmb[j])) for j in 1:3])
    @test dim.(siteinds(ops3)) == [4, 4, 4]      # vectorized physical legs
    @test all(dim.(linkinds(ops3)) .> 1)          # genuinely correlated, not a product

    @test isapprox(expval_LR_ops(ll, rr, ops3, b) / nrm3, ref3; rtol=1e-8)

    # --- k = 2 still matches the MPO-container spelling it replaced ---
    eps_mpo = ITransverse.epsilon_brick_ising(MPε)
    @test expval_LR_ops(ll, rr, eps_mpo, b) ≈
          expval_LR_ops(ll, rr, ITransverse._as_folded_operator(eps_mpo), b)

    # the Site leg must carry the folded physical dimension
    bad = MPS([settags(O3[j] * cmb[j], "Site", combinedind(cmb[j])) for j in 1:3])
    @test_throws ErrorException ITransverse.folded_tMPO_op(
        b, ITransverse._time_sites(rr, b), replaceind(bad[1], only(inds(bad[1], "Site")),
                                                      Index(7, "Site")))
end
