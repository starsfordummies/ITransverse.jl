using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: init_cone

@testset "Light cone initialization and tensors for Ising" begin
tp = ising_tp()
b = FoldtMPOBlocks(tp)
evs_X = ComplexF64[]
for nn = 1:6
    c0 = init_cone(b,nn)
    push!(evs_X, expval_LR(c0,c0,[0,1,1,0], b))
end

@test norm(evs_X - ITransverse.BenchData.bench_X_04_plus[1:6]) < 1e-12
end



@testset "Check that init_cone full vs triangular give the same result" begin
tp = ising_tp()
b = FoldtMPOBlocks(tp)
nn = 6
c0_full = init_cone(b, nn; full=true)
c0_tri = init_cone(b, nn; full=false)

#c0_alt = alt_init_cone(siteinds(c0), b, nn)

@test fidelity(c0_full,c0_tri) ≈ 1 
end

@testset "Check init_cone(mps, mpo)" begin

    tp = ising_tp()
    b = FoldtMPOBlocks(tp)
    nn = 6
    ts = siteinds(4, nn)
    psi = folded_tMPS(b, ts)
    m = folded_tMPO(b,ts)

    c0_mpsi = init_cone(psi, m)

    c0_tri = init_cone(b, nn; full=false)

    @test fidelity(c0_mpsi,c0_tri) ≈ 1 

end