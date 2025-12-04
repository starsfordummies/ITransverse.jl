using ITensors, ITensorMPS, ITransverse
using Test

using ITransverse.ITenUtils: applyd_l

@testset "contractd_l/applyd_l" begin
ss = siteinds(4, 16)

n_ext = 4 
oo= folded_tMPO_ext(FoldtMPOBlocks(ising_tp()), ss ; LR=:right, n_ext)
psi = random_mps(ss[1:end-n_ext], linkdims = 30)

opsi = applyd_l(oo,psi)
opsi_alt = applyn(oo,psi)

@test fidelity(opsi, opsi_alt) ≈ 1 

A = random_mpo(ss) + random_mpo(ss)
ψ = random_mps(ss, linkdims = 30)

opsi = applyd_l(A,ψ; cutoff=1e-12, maxdim=60)
opsi_alt = apply(A,ψ; cutoff=1e-12)
@test fidelity(opsi, opsi_alt) ≈ 1

@test ortho_lims(opsi) == 16:16



ss = siteinds(4, 40)

A = random_mpo(ss) + random_mpo(ss)
ψ = random_mps(ss, linkdims = 60)

opsi1 = applyd_l(A,ψ; cutoff=1e-12)
opsi2 = apply(A,ψ; cutoff=1e-12);
opsi3 = applyn(A,ψ; cutoff=1e-12);

@test fidelity(opsi1, opsi2) ≈ 1
@test fidelity(opsi2, opsi3) ≈ 1

end
