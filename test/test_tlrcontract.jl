using ITensors, ITensorMPS, ITransverse
using Test


ss = siteinds("S=1/2", 12)
ss2 = siteinds("S=1/2", 16)
ss2[1:length(ss)] = ss

ψL = random_mps(ComplexF64, ss, linkdims=18) 
ψR = random_mps(ComplexF64, ss, linkdims=16) 

AL = random_mpo(ss2) + im*random_mpo(ss2)
AR = random_mpo(ss2) + im*random_mpo(ss2)


for kk = length(ss)+1:length(ss2)
    AR[kk] *= ITensor([1,0],ss2[kk])
    AL[kk] *= ITensor([1,0],ss2[kk]')
end

LO = applyns(AL, ψL; truncate=false)
OR = applyn(AR, ψR; truncate=false)

cutoff = 1e-20
maxdim=128

llt, rrt, sst = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:left)
llt_left, rrt_left, sst_left = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:right)

llt2, rrt2, sst2 = ITransverse.truncate_rsweep_rtm(LO, OR; cutoff, maxdim)

ll, rr, ss = ITransverse.tlrcontract_old(ψL, AL, AR, ψR; cutoff, maxdim)
abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr))/abs(gen_fidelity(llt,rrt))
#@test abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr)) < 1e-5

llc, rrc, ssc = tlrapply(ψL, AL, AR, ψR; alg="naiveRTM", cutoff, maxdim, direction=:left)

#@code_warntype  tlrapply(ψL, AL, AR, ψR; cutoff, maxdim)

@test fidelity(ll, llc) > 0.9999
@test fidelity(ll, llt) > 0.9999
@test fidelity(llt, llt2) > 0.9999
@test fidelity(llc, llt2) > 0.95
@test fidelity(rrc, rrt2) > 0.95

@test fidelity(llc, llt) ≈ 1 
@test fidelity(rrc, rrt) ≈ 1 

fidelity(rr, rrc)
fidelity(rr, rrt)




ss = siteinds("S=1/2", 40)

ψL = random_mps(ComplexF64, ss, linkdims=100) 
ψR = random_mps(ComplexF64, ss, linkdims=120) 

AL = random_mpo(ss) + im*random_mpo(ss)
AR = random_mpo(ss) + im*random_mpo(ss)

ov = overlap_noconj(ψL,ψR)
ψL = ψL / sqrt(ov)
ψR = ψR / sqrt(ov)

cutoff = 1e-60
maxdim=256
mindim=256
truncp = (; cutoff, maxdim)

rtmr = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="RTM", truncp..., direction=:right)
rtmr.ov_before

rtml = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="RTM", truncp..., direction=:left)

rtmnl = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="naiveRTM", truncp..., direction=:left)
rtmnr = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="naiveRTM", truncp..., direction=:right)

fidelity(rtmr.L,rtmnr.L) 
fidelity(rtml.L,rtmnr.L) 
fidelity(rtmnl.L,rtmnr.L) 


fidelity(rtmr.R,rtmnr.R) 
fidelity(rtml.R,rtmnr.R) 
fidelity(rtmnl.R,rtmnr.R) 

overlap_noconj(rtmr)
overlap_noconj(rtml)
overlap_noconj(rtmnr)
overlap_noconj(rtmnl)

@test fidelity(rtmr.L,rtmnr.L) > 0.99
@show fidelity(rtmr.R, rtmnl.R) #> 0.99

@test (abs(gen_fidelity(rtmr.L, rtmr.R) - gen_fidelity(rtmnr.L, rtmnr.R)))/ abs(gen_fidelity(rtmr.L, rtmr.R)) < 0.1


cutoff = 1e-20
maxdim=256
truncp = (; cutoff, maxdim)
left, right, s = ITransverse.trapply(ITensors.Algorithm("RTM"), ψL, AR, ψR; truncp..., direction=:right)
leftn, rightn, sn = ITransverse.trapply(ITensors.Algorithm("naiveRTM"), ψL, AR, ψR; direction=:right)
leftref, rightref, sref = ITransverse.trapply(ITensors.Algorithm("densitymatrix"), ψL, AR, ψR; direction=:left)

@test siteinds(right) == siteinds(rightn)
@test siteinds(right) == siteinds(rightref)
@test siteinds(leftref) == siteinds(rightref)




ss = siteinds("S=1/2", 40)

ψL = random_mps(ComplexF64, ss, linkdims=100) 
ψR = random_mps(ComplexF64, ss, linkdims=120) 

AL = random_mpo(ss) + im*random_mpo(ss)
AR = random_mpo(ss) + im*random_mpo(ss)

truncp = (; cutoff=1e-5, maxdim=100)

ref = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="notrunc", truncp..., direction=:right)

overlap_noconj(ref)
rtmr = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="RTM", truncp..., direction=:right)
rtml = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="RTM", truncp..., direction=:left)

overlap_noconj(rtmr)
overlap_noconj(rtml)

overlap_noconj(ref)/overlap_noconj(rtml)
rtml.ov_before
overlap_noconj(ref)/overlap_noconj(rtmr)
rtmr.ov_before


rtmnr = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="naiveRTM", truncp..., direction=:right)
rtmnl = ITransverse.tlrapply(ψL, AL, AR, ψR; alg="naiveRTM", truncp..., direction=:left)

overlap_noconj(rtmnr)
overlap_noconj(rtmnl)

overlap_noconj(ref)/overlap_noconj(rtmnl)
rtmnl.ov_before
overlap_noconj(ref)/overlap_noconj(rtmnr)
rtmnr.ov_before

overlap_noconj(rtmnr)
overlap_noconj(rtmnl)

overlap_noconj(ref)
rtmr = ITransverse.trapply(ψL, AR, ψR; alg="RTM", truncp..., direction=:right)
rtml = ITransverse.trapply(ψL, AR, ψR; alg="RTM", truncp..., direction=:left)
rtmr.ov_before
overlap_noconj(rtmr)
overlap_noconj(ψL, rtmr.R)
