using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: up_state, vZ, sym_left

# Light cone with "densitymatrix" truncation on the folded network with Z2 (SzParity) QNs.
# Integrable Ising (no longitudinal field) from |up>, measuring Z: both are in the even sector.
# The QN cone must reproduce the one built without QNs.

function qn_cone(qns::Bool; Nsteps=30)
    s = siteind("S=1/2"; conserve_szparity=qns)
    mp = IsingParams(1.0, 0.4, 0.0; phys_site=s)
    tp = tMPOParams(mp; dt=0.1, scheme=Murg(), nbeta=0, init_state=up_state)
    b = FoldtMPOBlocks(tp)

    cp = DoCheckpoint(joinpath(mktempdir(), "cp_qn_cone.jld2"); params=tp,
        f_obs=(Z = s -> expval_LR(s.L, s.R, vZ, s.b),),
        f_savestate=(L = s -> s.L, R = s -> s.R, b = s -> s.b))

    truncp = (; cutoff=1e-12, maxdim=64, direction=:right, alg="densitymatrix")
    cone_params = ConeParams(; truncp, opt_method=:sym, optimize_op=vZ)
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
