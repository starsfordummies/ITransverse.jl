using ITensors, ITensorMPS, ITransverse
using Test

# check that skewed apply/truncate works
# assume we are on the left edge of a light cone, so psiL is short, gets extended by AL, 
# whereas psiR is longer and gets shortened by AR 

preserve_mps_tags = true
cutoff = 1e-20
maxdim=128
direction = :right 
compute_overlaps = true
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



ss = siteinds("S=1/2", 30)

ψL = random_mps(ComplexF64, ss[1:27], linkdims=60)
ψR = random_mps(ComplexF64, ss, linkdims=102) 

AR = random_mpo(ss) + im*random_mpo(ss)

for jj = 28:30
    AR[jj] *= ITensor([1,0], ss[jj]')
end

truncp = (; cutoff=1e-8, maxdim=10, preserve_mps_tags=false)

ref = ITransverse.trapply(ITensors.Algorithm("notrunc"), ψL, AR, ψR)


rR = ITransverse.trapply(ITensors.Algorithm("RTM"), ψL, AR, ψR; truncp..., direction=:right)
rL = ITransverse.trapply(ITensors.Algorithm("RTM"), ψL, AR, ψR; truncp..., direction=:left)
nrL = ITransverse.trapply(ITensors.Algorithm("naiveRTM"), ψL, AR, ψR; truncp..., direction=:left)
nrR = ITransverse.trapply(ITensors.Algorithm("naiveRTM"), ψL, AR, ψR; truncp..., direction=:right)

overlap_noconj(ref)

overlap_noconj(rR)
overlap_noconj(rL)
overlap_noconj(nrL)
overlap_noconj(nrR)
