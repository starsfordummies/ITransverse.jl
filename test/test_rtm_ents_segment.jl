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

S2_manual = -log(rho2_manual)

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