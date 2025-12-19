
using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state

# Check that folded tMPO with identity on top reduces to identity 

Nsteps = 4

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")
time_sites_fold = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")

random_eh = expH_random_symm_svd_1o(0.9)
ss = firstsiteinds(random_eh)

tp = ising_tp()
ising_eh = build_Ut(ss, expH_ising_murg, tp.mp; dt=0.2)

init_state = (rand(2))
init_statef = kron(init_state,conj(init_state))

ITensors.state(::StateName"rand_prod", ::SiteType"S=1/2") = init_state
# Temporal contraction 

init_psi = productMPS(ss, "rand_prod")
init_rho = outer(dag(init_psi)', init_psi)

norm(init_rho)

# Check1 : U Udag reduces to identity 

rho0 = kron(init_state, init_state)

b= FoldtMPOBlocks(random_eh; init_state=rho0)

mpo_fold =         folded_tMPO(b, time_sites_fold)
left_fold =   folded_left_tMPS(b, time_sites_fold)
right_fold = folded_right_tMPS(b, time_sites_fold)

m1 = apply(mpo_fold, right_fold)
m1 = overlap_noconj(m1, left_fold)

overlap = expval_LR(left_fold, mpo_fold, right_fold)

@test norm(init_rho) ≈ m1 ≈ overlap

# What if only on first and last sites 

s2 = siteinds("S=1/2",2)
init_psi = productMPS(s2, "rand_prod")
init_rho = outer(dag(init_psi)', init_psi)

@test norm(init_rho) ≈ overlap_noconj(left_fold, right_fold)