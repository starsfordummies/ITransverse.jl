using ITensors, ITransverse
using ITransverse: check_gencan_left, check_gencan_right
using Test

test_linkdim= 40 
test_chimax = 40 
s = siteinds(4, 50)

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
rr = ll + dag(ll)

# TODO checks for symmetric forms 

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
llc = deepcopy(ll)

lln = gen_canonical_right(ll)
@test check_gencan_right(lln, lln)

lln= gen_canonical_left(ll)
@test check_gencan_left(lln, lln)

#test we don't touch the original 
@test llc[5] == ll[5] 

lln