using ITensors, ITensorMPS
using ITransverse 
using Test


@testset "Generalized SVD entropies (symmetric/no symm)" begin
s = siteinds(4, 20)

psi = random_mps(ComplexF64, s, linkdims=40)

sgen_sv_s = generalized_svd_vn_entropy_symmetric(psi)

sgen_sv = generalized_svd_vn_entropy(psi,psi)

@test sgen_sv_s ≈ sgen_sv

end