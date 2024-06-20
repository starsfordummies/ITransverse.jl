@info "Check that we can build a transverse ising folded tMPS by contracting a tMPO tensor with [1,0,0,0] "

using ITensors
using ITransverse
using Test


JXX = 1.0  
hz = 0.4
gx = 0.0
dt = 0.1

nbeta=0

init_state = plus_state

sigZ = ComplexF64[1,0,0,-1]

Nsteps = 40

time_sites = siteinds("S=3/2", Nsteps)

mp = model_params("S=1/2", JXX, hz, gx, dt)
tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_parallel_field_murg, mp, nbeta, init_state)


zero_state = Vector{ComplexF64}([1,0])

Nsteps = 10

mpo_X = build_folded_tMPO(tp, sigX, time_sites)
left_mps = build_folded_left_tMPS(tp, time_sites)

w2 = mpo_X[2]
a2 = left_mps[2]

close_ten = ITensor([1,0,0,0], ind(w2,2))

w2c = close_ten * w2

a2arr = array(a2)
w2arr = array(w2c)
w2arr = permutedims(w2arr,(1,3,2))

@test a2arr/norm(a2arr) ≈ w2arr/norm(w2arr)

