using ITensors, ITensorMPS, ITransverse
using ITransverse: plus_state, up_state



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
    return timeEvo_MPO_2ndOrder(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;)[:]
  else
    return timeEvo_MPO_2ndOrder_LRflipped(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;)[:]
  end
end

bulk_evo = hcat(
  map(
    nt -> fill_bulk_evolution(nt, s, dt, Jxx, λ, hz, gx),
    range(1,N_t)
  )...
)

input = hcat(
  ψ0[:],
  bulk_evo,
  ψ0[:]
)

L, TL, TR, R = construct_unfolded_tMPS_tMPO(input)


inner(L,R)

L2 = apply(TL, L)

R2 = apply(TR, R)

inner(L2,R2)

norm(L2)

norm(R2)