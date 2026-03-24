using ITensors, ITensorMPS, ITransverse
using Test

ss = siteinds("S=1/2", 20)
ss2 = siteinds("S=1/2", 24)
ss2[1:length(ss)] = ss
ss
ss2
ψL = random_mps(ComplexF64, ss, linkdims=32) 
ψR = random_mps(ComplexF64, ss, linkdims=50) 

AL = random_mpo(ss2) + im*random_mpo(ss2)
AR = random_mpo(ss2) + im*random_mpo(ss2)

for kk = length(ss)+1:length(ss2)
    AR[kk] *= ITensor([1,0],ss2[kk])
    AL[kk] *= ITensor([1,0],ss2[kk]')
end

LO = applyns(AL, ψL; cutoff=1e-12, maxdim=200)
OR = applyn(AR, ψR; cutoff=1e-12, maxdim=200)

llt, rrt, sst = ITransverse.truncate_sweep(LO, OR; cutoff=1e-10, maxdim=40, direction="right")

ll, rr, ss = ITransverse.tlrcontract(ψL, AL, AR, ψR; cutoff=1e-10, maxdim=40)
 abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr))
@test abs( gen_fidelity(llt,rrt) - gen_fidelity(ll,rr)) < 1e-5
