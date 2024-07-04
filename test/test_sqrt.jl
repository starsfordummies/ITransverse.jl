using LinearAlgebra
using ITensors
using ITransverse
using ITransverse.ITenUtils
using Test

@testset "Testing sqrt of matrix decompositions" begin
    iL = Index(40)
    iR = sim(iL)

    a = random_itensor(ComplexF64, iL, iR)
    a /= norm(a)
    a = symmetrize(a)

    iTemp = sim(iR)

    sqa = sqrt(a, (iL, iTemp))
    sqa2 = sqrt(a, (iTemp, iR))

    @test sqa * sqa2 ≈ a

end