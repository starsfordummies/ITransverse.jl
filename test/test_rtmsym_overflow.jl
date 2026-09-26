using ITensors, ITensorMPS
using ITransverse
using Test

# A column whose local tensors all carry a large scalar factor has the same fixed point, but its
# eigenvalue grows like factor^N. The RTMsym sweep used to build unnormalised environments and the
# power method rescaled the wrong site, so such columns overflowed into NaN inside the SVD.
@testset "RTMsym power method: no overflow for large-norm columns" begin
    tp = tMPOParams(IsingParams(1.0, 1.0, 0.0); dt=0.1, dbeta=-0.1im, scheme=Murg(), nbeta=8, init_state=up_state)
    b = FwtMPOBlocks(tp)
    s = addtags(siteinds("S=1/2", 88; conserve_qns=false), "time")
    mpo = fw_tMPO(b, s, tr=tp.bl); psi0 = fw_tMPS(b, s; tr=tp.bl, LR=:right)
    big = copy(mpo); for j in eachindex(big); big[j] = 50 * big[j]; end    # eigenvalue × 50^88 ≈ 1e150
    pm = PMParams(; truncp=(; cutoff=1e-10, maxdim=32, direction=:right, alg="RTMsym"), itermax=15, itermin=15,
                  eps_converged=1e-14, maxdims=[32], cutoffs=[1e-10], normalization="overlap", stuck_after=1000)
    ψa, _ = powermethod_sym(psi0, mpo, pm)
    ψb, _ = powermethod_sym(psi0, big, pm)
    Sa = gensym_renyi_entropies(unsided(ψa)).S1
    Sb = gensym_renyi_entropies(unsided(ψb)).S1
    @test all(isfinite, Sb)
    # identical up to rounding: the scale bookkeeping (exp/log) changes the last bits, which the
    # still-unconverged iteration amplifies to ~5e-9 after 15 steps (also for an exact factor 2)
    @test maximum(abs.(Sa .- Sb)) < 1e-7
end
