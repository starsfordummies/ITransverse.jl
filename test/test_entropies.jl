using Test

using ITensors, ITensorMPS
using ITransverse 


@testset "Entropies related tests" begin
s = siteinds(4, 20)
psi = random_mps(s,linkdims=12)
psic = deepcopy(psi)
s_vn = vn_entanglement_entropy(psi)

# check that vn_entanglement_entropy() does not modify the psi 
@test psi[5] ≈ psic[5]
@test psi[9] ≈ psic[9]
@test psi[20] ≈ psic[20]

end
