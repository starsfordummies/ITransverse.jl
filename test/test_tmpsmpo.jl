@info "Check that we can build a transverse ising folded tMPS by contracting a tMPO tensor with [1,0,0,0] "

using ITensors
using ITransverse
using Test

zero_state = Vector{ComplexF64}([1,0])

JXX = 1.0  
hz = 0.4
dt = 0.1
nbeta = 0.0

Nsteps = 10
params = pparams(JXX, hz, dt, nbeta, zero_state)

sigX = ComplexF64[0,1,1,0]

time_sites = siteinds("S=3/2", Nsteps)

init_mps = build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

mpo_X = build_ising_folded_tMPO(build_expH_ising_murg, params, sigX, time_sites)

w2 = mpo_X[2]
a2 = init_mps[2]

close_ten = ITensor([1,0,0,0], ind(w2,2))

w2c = close_ten * w2

a2arr = array(a2)
w2arr = array(w2c)
w2arr = permutedims(w2arr,(1,3,2))

@test a2arr/norm(a2arr) ≈ w2arr/norm(w2arr)

