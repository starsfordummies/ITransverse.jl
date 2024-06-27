using Revise
using ITensors, ITransverse
using Test

s = siteinds("S=3/2", 20)

ll = random_mps(ComplexF64, s, linkdims=40)
llc = deepcopy(ll)

psi_gauged = gen_canonical_right(ll)

inner(llc, ll)
inner(llc, psi_gauged)/norm(psi_gauged)

check_gencan_right(psi_gauged, psi_gauged)

