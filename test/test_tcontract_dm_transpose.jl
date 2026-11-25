using ITensors, ITensorMPS, ITransverse
using Random, Test

# tcontract(densitymatrix) on a ψ whose site legs sit on the MPO's PRIMED legs applies the
# transpose (contract ψ' with A, output on A's unprimed legs). The conjugate MPO used to be
# built with a fixed replaceprime(3 => 1), which assumes ψ on the unprimed legs: the
# environments then never contracted (AssertionError on ndims(rho) after building
# 15-20-index tensors). block_transfer_eigs applies Eᵀ this way.
@testset "tcontract densitymatrix: A ψ and Aᵀ ψ" begin
    Random.seed!(11)
    dm = ITensors.Algorithm("densitymatrix")

    s = siteinds(2, 6)
    A = random_mpo(s) + random_mpo(s)
    ψ = random_mps(ComplexF64, s; linkdims=3)

    r, _ = tapply(dm, A, ψ; cutoff=1e-15, maxdim=100)
    ex = noprime(contract(A, ψ))
    @test norm(r - ex) / norm(ex) < 1e-12

    ψp = prime(siteinds, ψ)
    r, _ = tapply(dm, A, ψp; cutoff=1e-15, maxdim=100)
    ex = contract(A, ψp)
    @test siteinds(r) == siteinds(ex)
    @test norm(r - ex) / norm(ex) < 1e-12

    # QN: a left vector lives in the dual space (In arrows, here dag of a real state); primed, it
    # contracts H's Out-pointing primed legs
    sq = siteinds("S=1/2", 6; conserve_qns=true)
    os = OpSum()
    for j in 1:5
        os += 0.7, "S+", j, "S-", j+1
        os += 1.3, "S-", j, "S+", j+1
        os += "Sz", j, "Sz", j+1
    end
    H = MPO(os, sq)
    ϕ = random_mps(sq, [isodd(j) ? "Up" : "Dn" for j in 1:6]; linkdims=4)
    for v in (ϕ, prime(siteinds, dag(ϕ)))
        r, _ = tapply(dm, H, v; cutoff=1e-15, maxdim=100)
        ex = replaceprime(contract(H, v), 1 => 0)
        @test norm(r - ex) / norm(ex) < 1e-12
    end
end
