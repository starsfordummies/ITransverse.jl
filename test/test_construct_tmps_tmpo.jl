using ITensors, ITensorMPS, ITransverse
using Test

@testset "Another test for the tMPS-tMPO constructor, with QNs" begin

ss = siteinds("S=1/2",3,conserve_szparity=true)
psi_i = MPS(ss, "Up")
psi_f = (MPS(ss, "Up"))

oo1 = random_mpo(ss) + random_mpo(ss)
oo2 = random_mpo(ss) + random_mpo(ss)
oo3 = random_mpo(ss) + random_mpo(ss)

Uts = [oo1, oo2, oo3, oo2, oo3]

contraction1, _ = ITransverse.contract_tn_tetris(psi_i, Uts, psi_f)

psiL, Tc, psiR = ITransverse.construct_tMPS_tMPO(psi_i, Uts, psi_f)

TR = applyn(Tc, psiR)

 @test  siteinds(TR) == siteinds(psiL)
 @test  overlap_noconj(psiL, TR) ≈ contraction1 

end



@testset "Test tMPS-tMPO constructor for a finite system (with QNs)" begin

ss = siteinds("S=1/2", 7,conserve_szparity=true)
psi_i = MPS(ss, "Up")
psi_f = (MPS(ss, "Up"))

oo1 = random_mpo(ss) + random_mpo(ss)
oo2 = random_mpo(ss) + random_mpo(ss)
oo3 = random_mpo(ss) + random_mpo(ss)

Uts = [oo1, oo2, oo3, oo2, oo3]

contraction1, _  = ITransverse.contract_tn_tetris(psi_i, Uts, psi_f)

psiL, Tc, psiR = ITransverse.construct_tMPS_tMPO_finite(psi_i, Uts, psi_f)

TR = psiR 
length(Tc)

for T in reverse(Tc)
  TR = apply(T, TR, cutoff=1e-12)
end

 
@test siteinds(TR) == siteinds(psiL)
@test overlap_noconj(psiL, TR) ≈ contraction1 

end