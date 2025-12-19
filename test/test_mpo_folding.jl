using ITensors, ITensorMPS, ITransverse
using Test
using ITransverse: FoldITensor

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

tsf = siteinds(4, 7)
fold_ref = folded_tMPO(bfold, tsf)

vecfold2, combs = ITransverse.ITenUtils.vectorize_mpo(fold2)
vecfold_ref, combs2 = ITransverse.ITenUtils.vectorize_mpo(fold_ref)

@test fidelity(vecfold2, vecfold_ref) ≈ 1

foldMPS3, cPMPS3, cPsMPS3 = ITransverse.combine_and_fold(fwmps, fwmps; fold_op = [1 0;0 1], fold_init_state = [1,0,0,1], dag_W2=true)

clegs =  MPO(cPMPS1)


p1 = apply(clegs, foldMPS2)

p2 = ITransverse.reopen_inds!(copy(foldMPS2), clegs)

p3 = ITransverse.reopen_inds!(copy(foldMPS2), cPMPS1)


@test p1 ≈ p2
@test p1 ≈ p3


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

psik = copy(foldpsi)

for jj = 1:300
    psik = apply(fold2, psik; cutoff=1e-10, maxdim=32)
    normalize!(psik)
    @show maxlinkdim(psik)
end
