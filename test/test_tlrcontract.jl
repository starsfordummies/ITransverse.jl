using ITensors, ITensorMPS, ITransverse
using Test


ss = siteinds("S=1/2", 12)
ss2 = siteinds("S=1/2", 16)
ss2[1:length(ss)] = ss
ss
ss2
ψL = random_mps(ComplexF64, ss, linkdims=128) 
ψR = random_mps(ComplexF64, ss, linkdims=100) 

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

llt, rrt, sst = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:right)
llt_left, rrt_left, sst_left = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:left)

llt2, rrt2, sst2 = ITransverse.truncate_rsweep_rtm(LO, OR; cutoff, maxdim)

ll, rr, ss = tlrcontract_old(ψL, AL, AR, ψR; cutoff, maxdim)
 abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr))/abs(gen_fidelity(llt,rrt))
#@test abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr)) < 1e-5

llc, rrc, ssc = tlrapply(ψL, AL, AR, ψR; cutoff, maxdim)

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


cutoff = 1e-20
maxdim=256
direction = :right 
truncp = (; cutoff, maxdim, direction)
left, right, s = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp...)
leftll, rightrr, s = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; merge(truncp, (;direction=:left))...)

leftn, rightn, sn = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; direction=:left)
leftref, rightref, sref = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; direction=:left)

@show fidelity(left,leftn) 
@test fidelity(left,leftn) > 0.99
@show fidelity(right, rightn) #> 0.99

@test (abs(gen_fidelity(left, right) - gen_fidelity(leftn, rightn)))/ abs(gen_fidelity(left, right)) < 0.1
@test gen_fidelity(leftref, rightref) ≈ gen_fidelity(leftn, rightn) 


cutoff = 1e-20
maxdim=256
truncp = (; cutoff, maxdim, direction)
left, right, s = ITransverse.trapply(ITensors.Algorithm("RTM"), ψL, AR, ψR; truncp...)
leftn, rightn, sn = ITransverse.trapply(ITensors.Algorithm("naiveRTM"), ψL, AR, ψR; direction=:right)
leftref, rightref, sref = ITransverse.trapply(ITensors.Algorithm("densitymatrix"), ψL, AR, ψR; direction=:left)

@test siteinds(right) == siteinds(rightn)
@test siteinds(right) == siteinds(rightref)
@test siteinds(leftref) == siteinds(rightref)