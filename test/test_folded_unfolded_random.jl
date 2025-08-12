using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state


@testset "Testing that folded+projector is the same as amplitude^2 using transverse contraction using random spin 1/2 " begin

Ntime_steps = 20

nbeta = 0
Nsteps = nbeta + Ntime_steps + nbeta

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")
time_sites_fold = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")

random_eh = build_expH_random_symm_svd_1o(0.5)

init_state = normalize(rand(2))
init_statef = kron(init_state,conj(init_state))
Pz = [1,0,0,0]

ITensors.state(::StateName"rand_prod", ::SiteType"S=1/2") = init_state
# Temporal contraction 

init_prod = productMPS(firstsiteinds(random_eh), "rand_prod")
init_rho = outer(dag(init_prod)', init_prod)



b= FwtMPOBlocks(random_eh; init_state)
bf = FoldtMPOBlocks(random_eh, init_state=init_statef)


mpo_fw =          fw_tMPO(b, time_sites; tr = up_state)
left_mps =   fw_tMPS(b, time_sites; LR=:left, tr = up_state)
right_mps = fw_tMPS(b, time_sites; LR=:right, tr = up_state)

mpo_fold =         folded_tMPO(bf, time_sites_fold, fold_op = Pz)
left_fold =   folded_left_tMPS(bf, time_sites_fold, fold_op = Pz)
right_fold = folded_right_tMPS(bf, time_sites_fold, fold_op = Pz)

abs2.(overlap_noconj(left_mps, right_mps))
overlap_noconj(left_fold, right_fold) #< 1

@test abs( abs2.(overlap_noconj(left_mps, right_mps)) - overlap_noconj(left_fold, right_fold)) < 1e-5

mpo_fw_conj =          dag(fw_tMPO(b, time_sites; tr = up_state))

# Contract unfolded network 
ll = left_mps
rr = right_mps
for nn = 1:2
    ll = applyns(mpo_fw, ll)
    rr = applyn(mpo_fw, rr)
end
maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)

Lsq = abs2(overlap_noconj(ll,rr))

# Contract folded network 
ll = left_fold
rr = right_fold
for nn = 1:2
    ll = applyns(mpo_fold, ll)
    rr = applyn(mpo_fold, rr)
end
maxlinkdim(ll)
maxlinkdim(rr)

Lfold = overlap_noconj(ll,rr)

@test abs.(Lsq - Lfold)/Lsq < 1e-4

end