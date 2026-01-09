using ITensors, ITensorMPS, ITransverse
using Test
using ITransverse: FoldITensor
using LinearAlgebra

#@testset "MPO folding " begin

tp = ising_tp()
bfw = FwtMPOBlocks(tp)
bfold = FoldtMPOBlocks(tp)

ts = siteinds("S=1/2", 8)
fw = fw_tMPO(bfw, ts; tr=[1,0])
fw2 = fw_tMPO(bfw, ts; tr=[1,1])

fwmps = fw_tMPS(bfw,ts; tr=[1,1])

fold, cP, cPs = ITransverse.combine_and_fold(fw, fw; fold_op = nothing,  fold_init_state = nothing, dag_W2=true)
fold2, cP2, cPs2 = ITransverse.combine_and_fold(fw2, fw2; fold_op = [1 0 ; 0 1], fold_init_state=nothing, dag_W2=true)

foldMPS1, cPMPS1, cPsMPS1 = ITransverse.combine_and_fold(fwmps, fwmps; fold_op = [1 0;0 1], dag_W2=true)
foldMPS2, cPMPS2, cPsMPS2 = ITransverse.combine_and_fold(fwmps, fwmps; dag_W2=true)
foldMPS3, cPMPS3, cPsMPS3 = ITransverse.combine_and_fold(fwmps, fwmps; fold_op = [1 0;0 1], fold_init_state = [1,0,0,1], dag_W2=true)

tsf = siteinds(4, 7)
fold_ref = folded_tMPO(bfold, tsf)


vecfold2, combs = ITransverse.ITenUtils.vectorize_mpo(fold2)
vecfold_ref, combs2 = ITransverse.ITenUtils.vectorize_mpo(fold_ref)

@test fidelity(vecfold2, vecfold_ref) ≈ 1


tsf3 = siteinds(4, 6)
fold_ref3 = folded_tMPO(bfold, tsf3, rho0=ITensor([1,0,0,1], Index(4)))

fold3, cP3, cPs3 = ITransverse.combine_and_fold(fw2, fw2; fold_op = [1 0 ; 0 1], fold_init_state=[1,0,0,1], dag_W2=true)


vecfold3, combs = ITransverse.ITenUtils.vectorize_mpo(fold3)
vecfold_ref3, combs2 = ITransverse.ITenUtils.vectorize_mpo(fold_ref3)

@test fidelity(vecfold3, vecfold_ref3) ≈ 1



clegs =  MPO(cPMPS1)


p1 = MPO(apply(clegs, foldMPS2).data)

p2 = ITransverse.reopen_inds(foldMPS2, clegs)

p3 = ITransverse.reopen_inds(foldMPS2, cPMPS1)


@test p1 ≈ p2
@test p2 ≈ p3


ss = siteinds("S=1/2", 12)
oo = random_mpo(ss) + im*random_mpo(ss) 

ofold2, c1, c2 = ITransverse.combine_and_fold(oo, oo; fold_op=nothing, dag_W2=true)
ofold1 = ITransverse.folded_UUt(oo, new_siteinds=firstsiteinds(ofold2))
@test ofold1 ≈ ofold2




tp = ising_tp()
bfw = FwtMPOBlocks(tp)
bfold = FoldtMPOBlocks(tp)

ts = siteinds("S=1/2", 60)

fw = fw_tMPO(bfw, ts; tr=[1,0])
fw2 = fw_tMPO(bfw, ts; tr=[1,0])

fold, cP, cPs = ITransverse.combine_and_fold(fw, fw; fold_op = nothing,  fold_init_state = nothing, dag_W2=true)
fold2, cP2, cPs2 = ITransverse.combine_and_fold(fw2, fw2; fold_op = [1 0 ;0 1], fold_init_state=[1 0 ; 0 1], dag_W2=true)



fwpsi = fw_tMPO(bfw, ts; tr=[1,0])

foldpsi, cP2, cPs2 = ITransverse.combine_and_fold(fwpsi, fwpsi; fold_op = [1 0 ;0 1], fold_init_state=[1 0 ; 0 1], dag_W2=true)


Nsites = 8
# XXZ non-symmetric anywhere 

tp = tMPOParams(0.1,  ITransverse.ChainModels.expH_XXZ_2o, XXZParams(1.0, 0.8), 0, [1,0])
tp = tMPOParams(0.1,  expH_potts_murg, PottsParams(1.0, 0.8), 0, [1,0,0])

b_fw = FwtMPOBlocks(tp)
b_fold = FoldtMPOBlocks(tp)

ts = siteinds(dim(tp.mp.phys_site), Nsites)
fw = fw_tMPO(b_fw, ts; tr=[1,0,0])

fold, cP, cPs = ITransverse.combine_and_fold(fw, fw; fold_op  = [1 0 0; 0 1 0; 0 0 1],  fold_init_state = nothing, dag_W2=true)
fold_alt, cP, cPs = ITransverse.combine_and_fold(fw, dag(fw); fold_op  = [1 0 0; 0 1 0; 0 0 1],  fold_init_state = nothing, dag_W2=false)

tsf = siteinds(dim(tp.mp.phys_site)^2, Nsites-1)
fold_ref = folded_tMPO(b_fold, tsf)


vecfold, combs = ITransverse.ITenUtils.vectorize_mpo(fold)
vecfold_ref, combs = ITransverse.ITenUtils.vectorize_mpo(fold_ref)

@test fidelity(vecfold, vecfold_ref) ≈ 1
