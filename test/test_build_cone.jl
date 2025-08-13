using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse: init_cone

@testset "Light cone initialization and tensors for Ising" begin
tp = ising_tp()
b = FoldtMPOBlocks(tp)
evs_X = ComplexF64[]
for nn = 1:6
    c0, b = init_cone(tp,nn)
    push!(evs_X, expval_LR(c0,c0,[0,1,1,0], b))
end

@test norm(evs_X - ITransverse.ITenUtils.bench_X_04_plus[1:6]) < 1e-12
end



function alt_init_cone(ts, b, n)

    psi = folded_right_tMPS(b, ts)

    for jj = 2:n
        m = folded_tMPO(b,ts)
        psi = applyn(m, psi)
        orthogonalize!(psi, length(psi))
        orthogonalize!(psi, 1)
    end

    return psi, b
end


tp = ising_tp()
b = FoldtMPOBlocks(tp)
nn = 9
c0, b = init_cone(b, nn)
c0_alt, b = alt_init_cone(siteinds(c0), b, nn)

@test fidelity(c0,c0_alt) ≈ 1 