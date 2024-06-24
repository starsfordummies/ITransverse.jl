using LinearAlgebra
using ITensors
using ITransverse
using ITransverse.ITenUtils
using Test

@testset "Testing sqrt of matrix decompositions" begin
    iL = Index(40)
    iR = sim(iL)

    a = randomITensor(ComplexF64, iL, iR)
    a /= norm(a)
    a = symmetrize(a)

    iTemp = sim(iR)

    sqa = sqrt(a, (iL, iTemp))
    sqa2 = sqrt(a, (iTemp, iR))

    @test sqa * sqa2 ≈ a

    # SVD and Eigenvalues can fail for non positive def matrices


    # fsvd = svd(a, ind(a,1))
    # arec_svd = fsvd.U * fsvd.S * fsvd.V

    # @test norm(arec_svd - a) < 1e-12

    # sq_svd =  fsvd.U * sqrt(fsvd.S) * fsvd.V
    # @test norm(sq_svd - sqa) < 1e-12

    # feig = eigen(a, ind(a,1), ind(a,2))
    # arec_eig = feig.V * feig.D * (feig.Vt)
    # sq_eig =  feig.V * sqrt(feig.D) * feig.Vt
    # @test norm(arec_eig - a) < 1e-12


    # @test norm(sq_eig - sqa) < 1e-12

end