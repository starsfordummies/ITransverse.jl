using ITensors, ITensorMPS, ITransverse
using Test
# using ITransverse: plus_state, up_state
# using ITensorVisualizationBase, ITensorMakie, Makie, CairoMakie
# ITensorVisualizationBase.set_backend!("Makie");
# CairoMakie.activate!(type = "svg")


# read_for_print_theme = Theme(
#   size = (800,700),
#   figure_padding = 0,#(0,0,0,0)
# )
# set_theme!(read_for_print_theme);

# @show ITensorVisualizationBase.get_backend()


s = siteinds("S=1/2", 4, conserve_szparity=false)
sqn = siteinds("S=1/2", 4, conserve_szparity=true)


# ψ0 = MPS(s, "Up")

# ψ_rand = randomMPS(s, "Up", 5)

Jxx = 1.0
λ = 0.0
hz = 1.0
gx = 0.0

dt = 0.1
N_t = 12


function fill_bulk_evolution(nt, sites, dt, Jxx, λ, hz, gx)
  if iseven(nt)
    return timeEvo_MPO_2ndOrder(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;).data # return vector of ITensors rather than MPO
  else
    return timeEvo_MPO_2ndOrder_LRflipped(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;).data # return vector of ITensors rather than MPO
  end
end

function fill_bulk_evolution_MPO(nt, sites, dt, Jxx, λ, hz, gx)
  if iseven(nt)
    return timeEvo_MPO_2ndOrder(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;) # return vector of ITensors rather than MPO
  else
    return timeEvo_MPO_2ndOrder_LRflipped(sites, ["Id"], [λ], ["X"], [1.0], ["X"], [-Jxx], ["Z", "X"], [-hz, -gx], dt;) # return vector of ITensors rather than MPO
  end
end

function fill_bulk_symmetric(sites, dt, Jxx, hz, gx)
  tp1 = tMPOParams(dt,  ITransverse.ChainModels.build_expH_ising_murg_new, IsingParams(Jxx, hz, gx), 0, ITransverse.up_state)
 return tp1.expH_func(sites, tp1.mp, dt)
end

#ITensorVisualizationBase.visualize(ψ::AbstractMPS, args...; kwargs...) = ITensorVisualizationBase.visualize(ITensorMPS.data(ψ), args...; kwargs...)


ψ0qn = MPS(sqn, "Up")

L, TL, TR, R = construct_tMPS_tMPO(
  ψ0qn,
  map(
    nt -> fill_bulk_evolution_MPO(nt, sqn, dt, Jxx, λ, hz, gx),
    range(1, N_t)
  ),
  ψ0qn,
)

ψ0 = MPS(s, "Up")

L_symm, TL_symm, TR_symm, R_symm = construct_tMPS_tMPO(
  ψ0,
  map(
    nt -> fill_bulk_symmetric(s, dt, Jxx, hz, gx),
    range(1, N_t)
  ),
  ψ0,
)


# @visualize L edge_labels=(tags=true,);
# @visualize R edge_labels=(tags=true,);

length(L)

norm(L)

norm(R)

norm(L_symm)
norm(R_symm)

abs(overlap_noconj(L , R))

L2 = apply(ITensors.Algorithm("naive"), TL, L; truncate=false)

R2 = apply(ITensors.Algorithm("naive"), TR, R; truncate=false)

# @visualize figL2 L2 edge_labels=(tags=true,);
# figL2

# @visualize figR2 R2 edge_labels=(tags=true,);
# figR2

# save("visual_L2.pdf",  figL2)
# save("visual_R2.pdf",  figR2)

L2_symm = apply(ITensors.Algorithm("naive"), TL_symm, L_symm; truncate=false)

R2_symm = apply(ITensors.Algorithm("naive"), TR_symm, R_symm; truncate=false)

norm(L2)

norm(R2)


norm(L2_symm)

norm(R2_symm)



# we find the norm to be very similar, so the new tMPS states may be more evenly 
# normed when it comes to the repeated application of TL and TR




### Compare with tdvp should give the same ?


function buildExpHTFI( 
  sites::Vector{<:Index};
  J::Real = 1.0,
  λ::Real = 0.5,
  hz::Real = 1.0,
  gx::Real = 0.0,
  kwargs...
)
  if abs(λ) > 1.0
    throw(ArgumentError("cannot implement exponential decay with base larger than 1, λ = $(λ)!"))
  end
  # link_dimension
  # d0 = dim(op(sites, "Id", 1), 1)
  link_dimension = 3
  startState = 3
  endState = 1

  N = length(sites)

  EType = eltype(union(λ, J))

  # generate "regular" link indeces (i.e. without conserved QNs)
  linkindices = hasqns(sites) ? [Index([QN() => 1, QN("SzParity",1,2) => 1, QN("SzParity",0,2) => 1], "Link,l=$(n-1)") for n in 1:N+1] : [Index(link_dimension, "Link,l=$(n-1)") for n in 1:N+1]

  H = MPO(sites)
  for n in 1:N
    # siteindex s
    s = sites[n]
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n])
    # right link index rl
    rl = linkindices[n+1]

    # init empty ITensor with
    H[n] = ITensor(EType, ll, dag(s), s', rl)
    # add both Identities as netral elements in the MPS at corresponding location (setelement function)
    H[n] += setelt(ll[startState]) * setelt(rl[startState]) * op(sites, "Id", n)
    H[n] += setelt(ll[endState]) * setelt(rl[endState]) * op(sites, "Id", n)
    # local nearest neighbour and exp. decaying interaction terms
    H[n] += setelt(ll[startState]) * setelt(rl[2]) * op(sites, "X", n)
    if !iszero(λ)
      H[n] += setelt(ll[2]) * setelt(rl[2]) * op(sites, "Id", n) * λ  # λ Id,  on the diagonal
    end
    H[n] += setelt(ll[2]) * setelt(rl[endState]) * op(sites, "X", n) * -J # Jxx σˣ
    if !iszero(hz) || !iszero(gx) 
      H[n] += setelt(ll[startState]) * setelt(rl[endState]) * (op(sites, "Z", n) * -hz + op(sites, "X", n) * -gx) # hz σᶻ
    end
  end
  # project out the left and right boundary MPO with unit row/column vector
  L = ITensor(linkindices[1])
  L[startState] = 1.0

  R = ITensor(dag(linkindices[N+1]))
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end


H_ising = buildExpHTFI(sqn;J=Jxx,λ,hz,gx)
# ψ0 = MPS(sqn,"Up")
psi_tdvp = tdvp(
    H_ising,
    -im*(N_t*dt),
    ψ0qn;
    time_step=-im*dt,
    maxdim=200,
    cutoff=1e-12,
    normalize=true,
    outputlevel=1,
)



tn_contraction_transverse = abs(overlap_noconj(L2, R2))
tn_contraction_transverse_symm = abs(overlap_noconj(L2_symm, R2_symm))
tn_contraction_tdvp = abs(inner(ψ0, psi_tdvp))
# abs(overlap_noconj(L2_symm , R2_symm))
@show abs(tn_contraction_tdvp - tn_contraction_transverse)
@test isapprox(tn_contraction_tdvp, tn_contraction_transverse; atol=1e-3 )