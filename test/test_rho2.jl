using ITensors, ITensorMPS
using ITransverse
using Test

ss = siteinds("S=1/2", 20)

psi = random_mps(ss, linkdims=60)


phi = random_mps(ss, linkdims=40)

@test rtm2_contracted(psi, phi, 12) ≈ ITransverse.rtm2_contracted_m(psi, phi, 12)
