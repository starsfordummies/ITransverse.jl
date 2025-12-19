using ITensors, ITensorMPS, ITransverse
using Test

ss = siteinds("S=1/2", 6, conserve_szparity=false)

psi = MPS(ss, "Up")
oo1 = build_Ut(ss, expH_ising_murg, 1, 0.8, 0.0; dt=0.2)
oo2 = build_Ut(ss, expH_ising_murg, 1, 1.5, 0.0; dt=0.2)
oo3 = build_Ut(ss, expH_ising_murg, 1, 0.4, 0.0; dt=0.2)


psi_i = normalize(apply(oo1, psi))
psi_f = normalize(apply(oo2, psi))
mpo_rows = [oo1,oo2,oo2,oo3]


tetris = ITransverse.contract_tn_tetris(psi_i, mpo_rows, psi_f)

left, cols, right = ITransverse.construct_tMPS_tMPO_finite(psi_i, mpo_rows, psi_f)

transverse = ITransverse.contract_tn_transverse(left, cols, right)

@test isapprox(tetris, transverse, rtol=1e-4)


# TODO with QN 