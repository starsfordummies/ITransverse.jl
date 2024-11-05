using ITensors, ITensorMPS
using Test

using ITransverse

ss = siteinds(4, 20)
psi = random_mps(ComplexF64, ss, linkdims=30)
phi = random_mps(ComplexF64, ss, linkdims= 20)
o = random_mpo(ss) + random_mpo(ss) + random_mpo(ss) 

# @info expval_LR(psi, o, phi)
# @info ITransverse.expval_LR_apply(psi,o, phi)

@test expval_LR(psi, o, phi) ≈ ITransverse.expval_LR_apply(psi,o, phi)
