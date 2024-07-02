using Revise
using ITensors, ITransverse
using Test

s = siteinds(4, 50)

ll = random_mps(ComplexF64, s, linkdims=40)
rr = ll + dag(ll)

lln, rrn = truncate_rsweep(ll, rr, cutoff=1e-12, chi_max=100)

@test check_gencan_right(lln, rrn)


ll = random_mps(ComplexF64, s, linkdims=40)
rr = ll + dag(ll)

lln, rrn = truncate_lsweep(ll, rr, cutoff=1e-12, chi_max=100)

@test check_gencan_left(lln, rrn)

# llc = deepcopy(ll)

# psi_gauged = gen_canonical_right(ll)

# inner(llc, ll)
# inner(llc, psi_gauged)/norm(psi_gauged)

# check_gencan_right(psi_gauged, psi_gauged)

