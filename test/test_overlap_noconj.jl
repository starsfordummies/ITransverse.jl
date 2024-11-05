using ITensors, ITensorMPS
using Test

using ITransverse

ss = siteinds(4, 20)
psi = random_mps(ComplexF64, ss, linkdims=30)
phi = random_mps(ComplexF64, ss, linkdims= 20)

@test overlap_noconj(psi, phi) ≈ ITransverse.overlap_noconj_ite(psi,phi)
