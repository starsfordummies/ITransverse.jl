using ITensors, ITensorMPS, ITransverse
using Test

@testset "Construct transverse MPS/MPO from spatial ones" begin
ss = siteinds("S=1/2", 6, conserve_szparity=false)

psi = MPS(ss, "Up")
oo1 = build_Ut(ss, expH_ising_murg, IsingParams(1, 0.8, 0.0); dt=0.2)
oo2 = build_Ut(ss, expH_ising_murg, IsingParams(1, 1.5, 0.0); dt=0.2)
oo3 = build_Ut(ss, expH_ising_murg, IsingParams(1, 0.4, 0.2); dt=0.2)


psi_i = normalize(apply(oo1, psi))
psi_f = normalize(apply(oo2, psi))
mpo_rows = [oo1,oo2,oo2,oo3]


tetris, chi_tetris = ITransverse.contract_tn_tetris(psi_i, mpo_rows, psi_f)

left, cols, right = ITransverse.construct_tMPS_tMPO_finite(psi_i, mpo_rows, psi_f)

transverse, chi_transverse  = ITransverse.contract_tn_transverse(left, cols, right)

@test isapprox(tetris, transverse, rtol=1e-4)
@show chi_tetris, chi_transverse

end


@testset "Construct transverse MPS/MPO from spatial ones (random)" begin
ss = siteinds("S=1/2", 9)

cutoff=1e-14
maxdim=256
psi = random_mps(ComplexF64, ss)
oo1 = random_mpo(ss) + im*random_mpo(ss)  
oo2 = random_mpo(ss) + im*random_mpo(ss)  
oo3 = random_mpo(ss) + im*random_mpo(ss)  

psi_i = normalize(apply(oo1, psi))
psi_f = normalize(apply(oo2, psi))
mpo_rows = [oo1,oo2,oo2,oo1,oo3]


tetris, chi_tetris = ITransverse.contract_tn_tetris(psi_i, mpo_rows, psi_f; cutoff,maxdim)

left, cols, right = ITransverse.construct_tMPS_tMPO_finite(psi_i, mpo_rows, psi_f)

transverse, chi_transverse  = ITransverse.contract_tn_transverse(left, cols, right; cutoff,maxdim)

@test isapprox(tetris, transverse, rtol=1e-4)
@show chi_tetris, chi_transverse

end
# TODO with QN 