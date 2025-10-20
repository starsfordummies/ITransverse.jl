using ITensors, ITensorMPS, ITransverse


s = siteinds("S=1/2", 4)


ψ0 = MPS(s, "Up")

Jxx = 1.0
λ = 0.2
hz = 1.0
gx = 0.0

dt = 0.1
N_t = 20

function fill_bulk_evolution(nt, sites, dt, Jxx, λ, hz, gx)
  if iseven(nt)
    return timeEvo_MPO_2ndOrder(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;).data # return vector of ITensors rather than MPO
  else
    return timeEvo_MPO_2ndOrder_LRflipped(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;).data # return vector of ITensors rather than MPO
  end
end

# hcat takes the vector of vectors (which is the ouput of map(...)) and stacks them in a matrix column by column
bulk_evo = hcat(
  map(
    nt -> fill_bulk_evolution(nt, s, dt, Jxx, λ, hz, gx),
    range(1, N_t)
  )...
)

# hcat the initial and final state (for the Loschmidt scenario)
input = hcat(
  ψ0[:],
  bulk_evo,
  ψ0[:]
)

# construct the temporal MPS for L and R, and the corresponding transfer matrices TL and TR
L, TL, TR, R = construct_unfolded_tMPS_tMPO(input)

norm(L)
norm(R)

inner(L, R)

L2 = apply(TL, L)

R2 = apply(TR, R)

inner(L2, R2)

norm(L2)

norm(R2)

# we find the norm to be very similar, so the new tMPS states may be more evenly 
# normed when it comes to the repeated application of TL and TR