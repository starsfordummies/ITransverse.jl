using ITensors, ITensorMPS
using ITransverse
using Test

""" Ideally here we'd like to truncate a left and a right MPS
in order to optimize their overlap. How close are the resulting two 
with respect to the original ones?
"""

@testset "Test various truncations" begin

chimaxs = 50
sites = siteinds("S=1/2", 50)
psiL = random_mps(ComplexF64, sites, linkdims=chimaxs)

# this is a hacky way to give an overlap ~ 1/sqrt(2)
psiR = normalize(dag(psiL) + psiL)

psiLc = deepcopy(psiL)
psiRc = deepcopy(psiR)

l, r, s, overlap = truncate_sweep_aggressive_normalize(psiL, psiR, cutoff=1e-10, chi_max=chimaxs, method="SVD")

# check if some inplace shenanigans have happened
@show inner(psiL,psiLc)
@show inner(psiR,psiRc)

# How much did we truncate L and R ? 
@show inner(psiL, l)/norm(l)
@show inner(psiR, r)/norm(r)

@show overlap_noconj(psiL,psiR) # This should be the maximum value!
@show overlap_noconj(l,r)/norm(l)/norm(r)

end
