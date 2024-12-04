using Test
using ITensors, ITensorMPS
using ITransverse 


@testset "Entropies related tests" begin
s = siteinds(4, 20)
psi = random_mps(s,linkdims=12)
psic = deepcopy(psi)
s_vn = vn_entanglement_entropy(psi)

# check that vn_entanglement_entropy() does not modify the psi 
@test psi[9] ≈ psic[9]

psi = deepcopy(psic)

eigen_rho = diagonalize_rdm(psi)

ents = build_entropies(eigen_rho, [1])

ents["S1"]
ents["S1n"]

@test ents["S1n"] ≈ s_vn


@test psi[9] ≈ psic[9]


end
