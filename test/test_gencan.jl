using ITensors, ITransverse
using ITransverse: check_gencan_left, check_gencan_right
using Test

test_linkdim= 40 
test_chimax = 40 
s = siteinds(4, 50)

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
rr = ll + dag(ll)

lln, rrn, ss = truncate_rsweep(ll, rr, cutoff=1e-12, chi_max=test_chimax, fast=false)

@test check_gencan_right(lln, rrn)


ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
rr = ll + dag(ll)

lln, rrn = truncate_lsweep(ll, rr, cutoff=1e-12, chi_max=test_chimax)

@test check_gencan_left(lln, rrn)


# TODO checks for symmetric forms 

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
llc = deepcopy(ll)

lln, ents = truncate_rsweep_sym(ll,  cutoff=1e-12, chi_max=test_chimax, method="SVD")
@test check_gencan_right(lln, lln)

lln, ents = truncate_rsweep_sym(ll,  cutoff=1e-12, chi_max=test_chimax, method="EIG")
@test check_gencan_right(lln, lln)

#test we don't touch the original 
@test llc[5] == ll[5] 

# Left sweeps 

ll = random_mps(ComplexF64, s, linkdims=test_linkdim)
llc = deepcopy(ll)
lln, ents, overlap = truncate_lsweep_sym(ll,  cutoff=1e-12, chi_max=test_chimax, method="SVD")
@test !any([hasplev(ii,1) for ii in siteinds(lln)])
@test check_gencan_left(lln, lln)

lln, ents, overlap = truncate_lsweep_sym(ll,  cutoff=1e-12, chi_max=test_chimax, method="EIG")
@test check_gencan_left(lln, lln)
@test !any([hasplev(ii,1) for ii in siteinds(lln)])


#test we didn't touch the original 
@test llc[5] == ll[5] 
