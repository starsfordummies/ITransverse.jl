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

cutoff = 1e-10
maxdim=120

llt, rrt, sst = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:right)
llt_left, rrt_left, sst_left = ITransverse.truncate_sweep(LO, OR; cutoff, maxdim, direction=:left)

llt2, rrt2, sst2 = ITransverse.truncate_rsweep_rtm(LO, OR; cutoff, maxdim)

ll, rr, ss = ITransverse.tlrcontract_old(ψL, AL, AR, ψR; cutoff, maxdim)
 abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr))/abs(gen_fidelity(llt,rrt))
#@test abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr)) < 1e-5

llc, rrc, ssc = ITransverse.tlrcontract(ψL, AL, AR, ψR; cutoff, maxdim)
llc, rrc, ssc = ITransverse.tlrapply(ψL, AL, AR, ψR; cutoff, maxdim)

fidelity(ll, llc)
fidelity(ll, llt)
fidelity(llt, llt2)
fidelity(llc, llt2)
fidelity(rrc, rrt2)

fidelity(llc, llt)
fidelity(rrc, rrt)

fidelity(rr, rrc)
fidelity(rr, rrt)

maxlinkdim(ll)
maxlinkdim(llt)
maxlinkdim(llt2)
maxlinkdim(llc)

llc
rrc


# @btime ITransverse.tlrcontract(ψL, AL, AR, ψR; cutoff, maxdim);

# @btime ITransverse.tlrcontract_old(ψL, AL, AR, ψR; cutoff, maxdim);

ss = siteinds("S=1/2", 40)

ψL = random_mps(ComplexF64, ss, linkdims=100) 
ψR = random_mps(ComplexF64, ss, linkdims=120) 

AL = random_mpo(ss) + im*random_mpo(ss)
AR = random_mpo(ss) + im*random_mpo(ss)


cutoff = 1e-10
maxdim=256
direction = :right 
truncp = (; cutoff, maxdim, direction)
left, right, s = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp...)
leftn, rightn, sn = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR;)

fidelity(left,leftn)
fidelity(right,rightn)

gen_fidelity(left, right)
gen_fidelity(leftn, rightn)

LO = applyns(AL, ψL; truncate=false)
OR = applyn(AR, ψR; truncate=false)

gen_fidelity(LO, OR)

#@btime ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp...);
#@btime ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; truncp...);