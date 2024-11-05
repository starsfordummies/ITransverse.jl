using ITensors, ITensorMPS
using Test

using ITransverse

ss = siteinds(4, 400)
psi = random_mps(ComplexF64, ss, linkdims=80)
phi = random_mps(ComplexF64, ss, linkdims= 55)

@test overlap_noconj(psi, phi) ≈ ITransverse.ITenUtils.overlap_noconj_ite(psi,phi)
