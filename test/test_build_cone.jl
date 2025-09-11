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




tp = ising_tp()
b = FoldtMPOBlocks(tp)
nn = 6
c0_full = init_cone(b, nn; full=true)
c0_tri = init_cone(b, nn; full=false)

#c0_alt = alt_init_cone(siteinds(c0), b, nn)

@test fidelity(c0_full,c0_tri) ≈ 1 