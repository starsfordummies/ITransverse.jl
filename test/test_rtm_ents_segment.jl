using Test

using ITensors, ITensorMPS
using ITransverse 


@testset "Generalized entropies (symmetric) for segment: contract vs orthogonalize&diagonalize" begin
    
ss = siteinds("S=1/2", 18)
psi = random_mps(ss, linkdims=26)

psi = psi/sqrt(overlap_noconj(psi,psi))

iA = 12
iB = 16

rho2_manual = ITransverse.gen_renyi2_sym_interval_manual(psi, iA,iB)

S2_manual = (rho2_manual)

psig = ITransverse.gen_canonical(psi, iA+2)

psigp = prime(linkinds, psig)
rhoc = ITensor(1)
for kk = iA:iB
    rhoc *= psig[kk]
    rhoc *= psigp[kk]
end

F = ITransverse.ITenUtils.symm_oeig(rhoc, (linkind(psig,iA-1), linkind(psig,iB)); cutoff=1e-13)
S2_diag = renyi_entropies(F.D.tensor.storage.data, which_ents=[2])["S2"][1]

@test S2_manual ≈ S2_diag

end


@testset "Generalized entropies (symmetric) for segment: contract vs orthogonalize&diagonalize" begin
    
LL = 10

ss = siteinds("S=1/2", LL)
psi = random_mps(ss, linkdims=19)

psi = psi/sqrt(overlap_noconj(psi,psi))

psip = prime(linkinds, psi)

iA = 4
fA = 8

rho = ITensor(1)

for kk = 1:iA-1
    rho *= psi[kk]
    rho *= psi[kk]'
end

for kk = iA:fA
    rho *= psi[kk]
    rho *= psip[kk]
end

for kk = fA+1:LL
    rho *= psi[kk]
    rho *= psi[kk]'
end

FA = ITransverse.ITenUtils.symm_oeig(rho, inds(rho,plev=0); cutoff=1e-13)

rho = ITensor(1)

for kk = 1:iA-1
    rho *= psi[kk]
    rho *= psip[kk]
end

for kk = iA:fA
    rho *= psi[kk]
    rho *= psi[kk]'
end

for kk = fA+1:LL
    rho *= psi[kk]
    rho *= psip[kk]
end

FB = ITransverse.ITenUtils.symm_oeig(rho, inds(rho, plev=1); cutoff=1e-13)

diag(FA.D) ≈ diag(FB.D)

end