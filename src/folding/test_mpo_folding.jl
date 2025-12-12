using ITensors, ITensorMPS, ITransverse
using Test

tp = ising_tp()
bfw = FwtMPOBlocks(tp)
bfold = FoldtMPOBlocks(tp)

ts = siteinds("S=1/2", 8)
fw = fw_tMPO(bfw, ts; tr=[1,0])
fw2 = fw_tMPO(bfw, ts; tr=[1,1])

fwmps = fw_tMPS(bfw,ts; tr=[1,1])

fold, cP, cPs = ITransverse.combine_fold_mpos(fw, fw; fold_op = nothing)
fold2, cP2, cPs2 = ITransverse.combine_fold_mpos(fw2, fw2; fold_op = [1,0,0,1])

foldMPS1, cPMPS1, cPsMPS1 = ITransverse.combine_fold_mpos(fwmps, fwmps; fold_op = [1,0,0,1])
foldMPS2, cPMPS2, cPsMPS2 = ITransverse.combine_fold_mpos(fwmps, fwmps; fold_op = nothing)

tsf = siteinds(4, 7)
fold_ref = folded_tMPO(bfold, tsf)

vecfold2, combs = ITransverse.ITenUtils.vectorize_mpo(fold2)
vecfold_ref, combs2 = ITransverse.ITenUtils.vectorize_mpo(fold_ref)

@test fidelity(vecfold2, vecfold_ref) ≈ 1

clegs =  MPO(cPMPS1)


p1 = apply(clegs, foldMPS1)

p2 = ITransverse.reopen_inds!(copy(foldMPS1), clegs)

p3 = ITransverse.reopen_inds!(copy(foldMPS1), cPMPS1)


@test p1 ≈ p2
@test p1 ≈ p3
