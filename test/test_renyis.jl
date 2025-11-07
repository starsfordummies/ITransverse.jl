using ITensors, ITensorMPS, ITransverse
using Test

@testset "Renyi entropies builders " begin

ss = siteinds("S=1/2", 20)
psi = random_mps(ComplexF64, ss, linkdims=40)

rs = renyi_entropies(psi, which_ents = [0.4, 1, 2, 4])

@test all(rs["S0.4"] .>= rs["S1.0"] )
@test all(rs["S1.0"] .>= rs["S2.0"] )
@test all(rs["S2.0"] .>= rs["S4.0"] )

end


#= Relations are prob not fulfilled here for complex spectra 
rs = gensym_renyi_entropies(psi, normalize_eigs=true)

plot(real(rs["S0.5"]))
plot!(real(rs["S1.0"]))
plot!(real(rs["S2.0"]))


plot(abs.(rs["S0.5"]))
plot!(abs.(rs["S1.0"]))
plot!(abs.(rs["S2.0"]))
=# 