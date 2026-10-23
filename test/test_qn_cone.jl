using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: up_state, vZ, sym_left

# Light cone with "densitymatrix" truncation on the folded network with Z2 (SzParity) QNs.
# Integrable Ising (no longitudinal field) from |up>, measuring Z: both are in the even sector.
# The QN cone must reproduce the one built without QNs.

function qn_cone(qns::Bool; Nsteps=30, alg="densitymatrix", opt_method=:sym)
    s = siteind("S=1/2"; conserve_szparity=qns)
    mp = IsingParams(1.0, 0.4, 0.0; phys_site=s)
    tp = tMPOParams(mp; dt=0.1, scheme=Murg(), nbeta=0, init_state=up_state)
    b = FoldtMPOBlocks(tp)

    cp = DoCheckpoint(joinpath(mktempdir(), "cp_qn_cone.jld2"); params=tp,
        f_obs=(Z = s -> expval_LR(s.L, s.R, vZ, s.b),),
        f_savestate=(L = s -> s.L, R = s -> s.R, b = s -> s.b))

    truncp = (; cutoff=1e-12, maxdim=64, direction=:right, alg)
    cone_params = ConeParams(; truncp, opt_method, optimize_op=vZ)
    ll, rr, cp = run_cone(init_cone(b), b, cone_params, cp, Nsteps)
    return ll, rr, cp.obs_hist[:Z]
end

@testset "QN light cone (densitymatrix, symmetric)" begin
    llq, rrq, zq = qn_cone(true)
    llp, rrp, zp = qn_cone(false)

    @test hasqns(rrq) && !hasqns(rrp)
    @test maximum(abs.(zq .- zp)) < 1e-8
    @test maxlinkdim(rrq) == maxlinkdim(rrp)

    # symmetric case: the left vector is a view of the right one, no data copied
    @test all(storage(llq[j]) === storage(rrq[j]) for j in eachindex(rrq))
    @test all(storage(llp[j]) === storage(rrp[j]) for j in eachindex(rrp))
    @test overlap_noconj(llq, rrq) ≈ 1
    @test overlap_noconj(sym_left(rrq), rrq) ≈ overlap_noconj(ITransverse.transpose_arrows(rrq), rrq)
end

# The single-MPS run_cone starts L and R as copies of the same state; with QNs the left one
# must be flipped to the dual arrows before the two-sided (tlrapply) paths can contract it.
@testset "QN light cone, two-sided truncation ($alg, $opt_method)" for (alg, opt_method) in
        (("RTM", :sym), ("RTM", :ns), ("densitymatrix", :ns))
    _, rrq, zq = qn_cone(true; Nsteps=14, alg, opt_method)
    _, _, zp = qn_cone(false; Nsteps=14, alg, opt_method)
    @test hasqns(rrq)
    @test maximum(abs.(zq .- zp)) < 1e-8
end

# Fused layout of a leg with a multi-dimensional QN block: charges are per state, not per block
@testset "_qn_fold_layout with multi-dimensional blocks" begin
    x = Index([QN("Q", 0) => 2, QN("Q", 1) => 1], "x")
    sp, perm = ITransverse._qn_fold_layout(x)
    q = [0, 0, 1]
    kron_charges = [q[a] - q[b] for b in 1:3 for a in 1:3]   # ket index fastest
    @test sort(perm) == 1:9
    @test reduce(vcat, [fill(val(c, "Q"), n) for (c, n) in sp]) == kron_charges[perm]
    @test flux(ITransverse._qn_fuser(x, Index(sp, "F"))) == QN("Q", 0)
end
