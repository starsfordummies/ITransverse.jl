using ITensors, ITensorMPS
using Test

using ITransverse

b = FoldtMPOBlocks(ising_tp())
ss = siteinds(4, 10)
psi = random_mps(ss[1:end-3], linkdims=20)
ooo = ITransverse.folded_tMPO_ext(b, ss; LR="R", n_ext=3, fold_op=[1,0,0,-1])

psi_ext = applyn(ooo,psi)

siteinds(psi_ext) == ss

phi = random_mps(ss[1:end-2], linkdims=20)

ooo = ITransverse.folded_tMPO_ext(b, ss; LR="L", n_ext=2, fold_op=[1,0,0,-1])

phi_ext = applyns(ooo,phi)

siteinds(phi_ext) == ss

ooo = ITransverse.folded_tMPO_ext(b, ss; LR="L", n_ext=4, fold_op=[1,0,0,-1])

rho = random_mps(ss, linkdims=20)

rho_cut = applyn(ooo,rho)
