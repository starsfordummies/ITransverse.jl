using ITensors, ITensorMPS, ITransverse
using ITransverse: plus_state, up_state

function Ising_MPO(sites::Vector{<:Index},JXX::Real,hz::Real,gx::Real)
    N = length(sites)
    os = OpSum()
    for j=1:N-1
        os += -JXX,"X",j,"X",j+1
        os += -hz,"Z",j
        os += -gx,"X",j
    end
    os += -hz,"Z",N
    os += -gx,"X",N
    return MPO(os,sites)
end


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
    if !iszero(hz)
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

@testset "Testing Generic time evolution operator in the non-transverse case" begin 
  cutoff = 1e-14
  maxdim = 128


  dt = 0.05
  n_iter =  Int(1.0/dt)

  JXX = 1.0
  hz = 1.0
  gx = 1.0

  N = 10
  ss = siteinds("S=1/2", N, conserve_szparity=false)
  state = fill("↑",N)
  psi0 = randomMPS(ComplexF64,ss,state,4)



  ##### Stefano's Ut
  mp1 = IsingParams(JXX, hz, gx)
  tp1 = tMPOParams(dt,  ITransverse.ChainModels.build_expH_ising_murg_new, mp1, 0, up_state)

  Ut1 = tp1.expH_func(ss, tp1.mp, dt)
  # Ut1 = build_expH_ising_murg_new2(ss, JXX, hz, gx, dt)

  psi_t1 = deepcopy(psi0)
  for tt = 1:n_iter
      psi_t1[:] = apply(Ut1, psi_t1; cutoff, maxdim, normalize=true)
  end

 

  ## Generic Ut MPO , note the different convention of signs
  Ut2 = timeEvo_MPO_2ndOrder(
      ss,
      ["Id"],
      [0.0],
      ["X"],
      [1.0],
      ["X"],
      [-JXX],
      ["Z", "X"],
      [-hz, -gx],
      dt;
  )


  psi_t2 = deepcopy(psi0)
  for tt = 1:n_iter
      psi_t2[:] = apply(Ut2, psi_t2; cutoff, maxdim, normalize=true)
  end






  ############
  # TDVP

  H_ising = Ising_MPO(ss,JXX,hz,gx)

  psi_tdvp = tdvp(
      H_ising,
      -1.0im,
      psi0;
      time_step=-im*dt,
      maxdim,
      cutoff,
      normalize=true,
      outputlevel=1,
  )


  # @show JXX
  # @show hz
  # @show gx
  # @show dt

  @test isapprox(abs(inner(psi_t1, psi_t2)),   1.0; rtol=5e-5)
  @test isapprox(abs(inner(psi_t1, psi_tdvp)), 1.0; rtol = 5e-5)
  @test isapprox(abs(inner(psi_t2, psi_tdvp)), 1.0; rtol = 5e-5)
end