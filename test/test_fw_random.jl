using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state, plus_state


#@testset "Testing that folded+projector is the same as amplitude^2 using transverse contraction using random spin 1/2 " begin

Ntime_steps = 4

Nsteps =  Ntime_steps 

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

random_eh = build_expH_random_symm_svd_1o(0.5)

init_state = normalize(rand(2))
#init_statef = kron((init_state),conj(init_state))

bf = FoldtMPOBlocks(random_eh, init_state=init_state)

b= FwtMPOBlocks(random_eh; init_state);

final_state = up_state
Pfinal = kron(final_state, final_state)


mpo_fw =    fw_tMPO(b, time_sites; tr = final_state)
left_mps =  fw_tMPS(b, time_sites; LR=:left, tr = final_state)
right_mps = fw_tMPS(b, time_sites; LR=:right, tr = final_state)

# Contract unfolded network 
ll = left_mps
rr = right_mps
rr = applyn(mpo_fw, rr)

maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)
expval_LR(left_mps, mpo_fw, right_mps)

expval_LR(dag(left_mps), dag(mpo_fw), dag(right_mps))



time_sites_fold = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")

mpo_fold =         folded_tMPO(bf, time_sites_fold, fold_op = Pfinal)
left_fold =   folded_left_tMPS(bf, time_sites_fold, fold_op = Pfinal)
right_fold = folded_right_tMPS(bf, time_sites_fold, fold_op = Pfinal)

abs2.(overlap_noconj(left_mps, right_mps))
overlap_noconj(left_fold, right_fold) ####


# N

Ntime_steps = 4

Nsteps =  Ntime_steps 

time_sites = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")

random_eh = ITransverse.ChainModels.build_expH_random()

init_state = normalize(rand(2))
#init_statef = kron((init_state),conj(init_state))

bf = FoldtMPOBlocks(random_eh, init_state=init_state)

b= FwtMPOBlocks(random_eh; init_state);

final_state = up_state
Pfinal = kron(final_state, final_state)


mpo_fw =    fw_tMPO(b, time_sites; tr = final_state)
left_mps =  fw_tMPS(b, time_sites; LR=:left, tr = final_state)
right_mps = fw_tMPS(b, time_sites; LR=:right, tr = final_state)

# Contract unfolded network 
ll = left_mps
rr = right_mps
rr = applyn(mpo_fw, rr)

maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)
expval_LR(left_mps, mpo_fw, right_mps)

expval_LR(dag(left_mps), dag(mpo_fw), dag(right_mps))



time_sites_fold = addtags(siteinds(16, Nsteps; conserve_qns=false), "time")

mpo_fold =         folded_tMPO(bf, time_sites_fold, fold_op = Pfinal)
left_fold =   folded_left_tMPS(bf, time_sites_fold, fold_op = Pfinal)
right_fold = folded_right_tMPS(bf, time_sites_fold, fold_op = Pfinal)

@test abs2.(overlap_noconj(left_mps, right_mps)) ≈ overlap_noconj(left_fold, right_fold) ####

@test abs2.(expval_LR(left_mps, mpo_fw, right_mps)) ≈ expval_LR(left_fold, mpo_fold, right_fold)

