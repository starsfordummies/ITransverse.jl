using ITensors, ITensorMPS, ITransverse
using Test

# check that skewed apply/truncate works
# assume we are on the left edge of a light cone, so psiL is short, gets extended by AL, 
# whereas psiR is longer and gets shortened by AR 

preserve_mps_tags = true
cutoff = 1e-20
maxdim=256
direction = :right 
truncp = (; cutoff, maxdim, preserve_mps_tags)

ss = siteinds("S=1/2", 10)


ψL = random_mps(ComplexF64, ss[1:5], linkdims=10)
ψR = random_mps(ComplexF64, ss, linkdims=12) 

AL = random_mpo(ss[1:8]) + im*random_mpo(ss[1:8])
AR = random_mpo(ss) + im*random_mpo(ss)

for jj = 6:8
AL[jj] *= ITensor([1,0], ss[jj]')
end

for jj = 9:10
    AR[jj] *= ITensor([1,0], ss[jj]')
end

lrs = ITransverse.tlrapply(ITensors.Algorithm("RTMold"), ψL, AL, AR, ψR; truncp...)
lrs2 = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp...)

# @test all(lrs .≈ lrs2)

lrnl = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; truncp..., direction=:left)

lrs3 = ITransverse.tlrapply(ITensors.Algorithm("RTM"), ψL, AL, AR, ψR; truncp..., direction=:left)
lrn = ITransverse.tlrapply(ITensors.Algorithm("naiveRTM"), ψL, AL, AR, ψR; truncp..., direction=:right)
