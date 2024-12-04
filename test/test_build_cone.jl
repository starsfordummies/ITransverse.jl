using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: init_cone

@testset "Light cone initialization and tensors for Ising" begin
tp = ising_tp()

evs_X = ComplexF64[]
for nn = 1:6
    c0, b = init_cone(tp,nn)
    push!(evs_X, expval_LR(c0,c0,[0,1,1,0], b))
end

@test norm(evs_X - ITransverse.ITenUtils.bench_X_04_plus[1:6]) < 1e-12
end

