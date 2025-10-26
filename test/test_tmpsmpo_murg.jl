using ITensors, ITensorMPS
using ITransverse
using Test


sigX = ComplexF64[0,1,1,0]

time_sites = siteinds(4, 10)

tp = ising_tp()

@info "Check that we can build a transverse ising folded tMPS by contracting a tMPO tensor with [1,0,0,0] "

b = FoldtMPOBlocks(tp)

zero_state = Vector{ComplexF64}([1,0])

Nsteps = 10

mpo_X = folded_tMPO(b, time_sites)
left_mps = folded_right_tMPS(b, time_sites; fold_op = [1,0,0,1])

w2 = mpo_X[2]
a2 = left_mps[2]

close_ten = ITensor([1,0,0,0], siteind(mpo_X,2))
w2c = w2 * close_ten
a2

# indices should be in the same ordering (site, R, L)
a2arr = array(a2)
w2arr = array(w2c)


@test a2arr/norm(a2arr) ≈ w2arr/norm(w2arr)

