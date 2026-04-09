using ITensors, ITransverse
using ITransverse: check_gencan_left, check_gencan_right, gen_canonical
using Test

test_linkdim= 40 
test_chimax = 40 
s = siteinds(4, 50)

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
rr = ll + dag(ll)

# TODO checks for symmetric forms 

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
llc = copy(ll)

lln = gen_canonical(ll, 1)
@test check_gencan_right(lln, lln)
@test !check_gencan_left(lln, lln)

lln = gen_canonical(ll, length(s))
@test !check_gencan_right(lln, lln)
@test check_gencan_left(lln, lln)

#test we don't touch the original 
@test llc[5] == ll[5] 
