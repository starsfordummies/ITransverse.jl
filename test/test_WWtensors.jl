using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state


#@testset "Testing that folded+projector is the same as amplitude^2 using transverse contraction using random spin 1/2 " begin

Nsteps = 20

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")
time_sites_fold = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")

random_eh = build_expH_random_symm_svd_1o(1)
ss = firstsiteinds(random_eh)

ising_eh = build_expH_ising_murg(ss, ising_tp().mp, 0.2)

init_state = normalize(rand(2))
init_statef = kron(init_state,conj(init_state))

ITensors.state(::StateName"rand_prod", ::SiteType"S=1/2") = init_state
# Temporal contraction 

init_psi = productMPS(ss, "rand_prod")
init_rho = outer(dag(init_psi)', init_psi)


# Check1 : U Udag reduces to identity 

inner(init_psi, init_psi)

Upsi = apply(random_eh, init_psi)
Upsi = apply(random_eh, Upsi)

inner(init_psi, Upsi)

Upsi = apply(swapprime(dag(random_eh), 0=>1), Upsi)
Upsi = apply(swapprime(dag(random_eh), 0=>1), Upsi)

inner(init_psi, Upsi)


Upsi = apply(random_eh, init_psi)
Upsi = apply(random_eh, Upsi)

inner(init_psi, Upsi)

Upsi = applyns((dag(random_eh)), Upsi)
Upsi = applyns((dag(random_eh)), Upsi)

inner(init_psi, Upsi)



# At the level of matrices, it should be a big identity

# Ising, works 
UUdag = apply(dag(ising_eh), ising_eh)
UUdagc = contract(UUdag)
UUdagc *= combiner(firstsiteinds(UUdag))
UUdagc *= combiner(firstsiteinds(UUdag)')
isid(UUdagc)

UUdag = apply(swapprime(dag(random_eh), 0=>1), random_eh)
UUdagc = contract(UUdag)
UUdagc *= combiner(firstsiteinds(UUdag))
UUdagc *= combiner(firstsiteinds(UUdag)')
matrix(UUdagc)
isid(UUdagc)


# Now check WW constructor 

WWl, WWc, WWr, (iL, iR, iP, iPs) = ITransverse.build_WW(random_eh)

row = WWl * WWc'' * delta(iL'', iR)
row = row * WWr'''' * delta(iR'', iL'''')
combP = combiner(iP, iP'', iP'''')
combPs = combiner(iP', iP''', iP''''')

row = row * ITransverse.vectorized_identity(iP')
row = row * ITransverse.vectorized_identity(iP''')
row = row * ITransverse.vectorized_identity(iP''''')

row = row * combP

# isid

row

# Check if they're vectorized identities 
vcomb = ITransverse.vectorized_identity(iP) * ITransverse.vectorized_identity(iP'') * ITransverse.vectorized_identity(iP'''')  
vcomb *= combP

row ≈ vcomb

