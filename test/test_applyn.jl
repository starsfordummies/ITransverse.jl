using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse.ITenUtils: check_mps_sanity

b = FoldtMPOBlocks(ising_tp())
ss = siteinds(4, 10)


n_ext = 3 

oooR = ITransverse.folded_tMPO_ext(b, ss; LR=:right, n_ext, fold_op=[1,0,0,-1])
@test count(isempty, siteinds(oooR,plev=0)) == n_ext
@test count(isempty, siteinds(oooR,plev=1)) == 0

oooL = ITransverse.folded_tMPO_ext(b, ss; LR=:left, n_ext, fold_op=[1,0,0,-1])
@test count(isempty, siteinds(oooL,plev=1)) == n_ext
@test count(isempty, siteinds(oooL,plev=0)) == 0


# Apply Right MPO to (shorter) Right tMPS: Extend 
psiR = random_mps(ss[1:end-n_ext], linkdims=20)

psiR_ext = applyn(oooR,psiR)
@test siteinds(psiR_ext) == ss
@test check_mps_sanity(psiR_ext)


# Apply Left tMPO to (shorter) Left tMPS: Extend
psiL = random_mps(ss[1:end-n_ext], linkdims=20)
psiL_ext = applyns(oooL,psiL)
@test check_mps_sanity(psiL_ext)

@test siteinds(psiL_ext) == ss


# Apply Right MPO to (equal length) Left tMPS: Chop 

psiL = random_mps(ss, linkdims=20)

psiL_chop = applyns(oooR,psiL)
@test check_mps_sanity(psiL_chop)
@test length(psiL_chop) == length(psiL)-n_ext

# Apply Left tMPO to (equal length) Right tMPS: Chop
psiR = random_mps(ss, linkdims=20)
psiR_chop = applyn(oooL,psiR)
@test check_mps_sanity(psiR_chop)
@test length(psiR_chop) == length(psiR)-n_ext


# What if they're only long 2 

ss = siteinds(4, 2)
psiR = random_mps(ss, linkdims=20)
psiR.llim
psiR.rlim
oooL = ITransverse.folded_tMPO_ext(b, ss; LR=:left, n_ext=1, fold_op=[1,0,0,1])

kk = applyn(oooL,psiR)


ITransverse.ITenUtils.contract_dangling!(kk)
kk.llim
kk.rlim
 orthogonalize(kk, length(kk))
 normalize(kk)


 #expvals
ss = siteinds(4, 10)

psiL = random_mps(ss, linkdims=20)
oooR = ITransverse.folded_tMPO_ext(b, ss; LR=:right, n_ext=3, fold_op=[1,0,0,-1])
psiR = random_mps(ss[1:end-3], linkdims=20)

expval_LR(psiL, oooR, psiR)