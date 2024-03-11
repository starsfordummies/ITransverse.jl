# author Jan Schneider

# Source: Van Damme, Haegeman, McCulloch, Vanderstraeten, arXiv:2302.14181 http://arxiv.org/abs/2302.14181
# and https://github.com/maartenvd/MPSKit.jl/blob/master/src/algorithms/timestep/timeevmpo.jl

# Given an MPO W of a Hamiltonian H in form 
#       | Id | 0 | 0  |
#  W =  |  C | A | 0  |
#       |  D | B | Id |

# then we find that the time evolution opertator U(t)  [Should be Eq.(45)]
# U(t) = exp(-ı t H) = exp(τ H) ≈
#   |Id + τD + τ^2/2 D^2 + τ^3/6 D^3 | τB + τ^2/2 {B D} + τ^3/6 {BDD}                  | τ^2 BB + τ^3/6{BBD}
# ≈ |C + τ^2/2 {CD} + τ^2/6 {CDD}    | A + τ/2({BC} + {AD}) + τ^2/6( {CBD} + {ADD})    | τ/2 {AB} + τ^2/6 ({ABD} + {BBC})
#   |CC + τ/3 {CCD}                  | {AC} + τ/3 ({ACD} + {CCB})                      | AA + τ/3 ({ABC} +{AAD})
# NB:
# {AB}  = AB + BA
# {ABB} = ABB + BAB + BBA
# {ABC} = ABC + ACB + BAC + BCA + CAB + CBA


using ITensors, ITensorTDVP, Printf

braket(A::ITensor, B::ITensor) = replaceprime(prime(A) * B + prime(B) * A, 2, 1)
braket(A::ITensor, B::ITensor, C::ITensor) = replaceprime(
    (A'' * B' * C) + (A'' * C' * B) + (B'' * A' * C) + (B'' * C' * A) + (C'' * A' * B) + (C'' * B' * A),
    3, 1)
braket2(A::ITensor, B::ITensor) = replaceprime((A'' * B' * B) + (B'' * A' * B) + (B'' * B' * A), 3, 1)

function timeEvoMPO2_Ising(
  sites,
  AString::String,
  JA::Number,
  BString::String,
  JB::Number,
  CString::String,
  JC::Number,
  DString::String,
  JD::Number,
  t;
  )

  # measure τ in imaginary units for actualy time evolution
  τ = -1.0im * t
  N = length(sites)
  U_t = MPO(N)

  link_dimension = 3
  # startState = 3
  # endState = 1

  linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

  # loop over real space finite-sized MPO and fill (here) homogeniously
  for n = 1:N
    # siteindex s
    s = sites[n]
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n])
    # right link index rl
    rl = linkindices[n+1]
    Id = op(sites, "Id", n)
    # A is possible exponential decay so test for "0"
    A = AString == "0" ? 0.0 * op(sites, "Id", n) : JA * op(sites, AString, n)
    B = JB * op(sites, BString, n)
    C = JC * op(sites, CString, n)
    D = JD * op(sites, DString, n)
    # Init ITensor inside MPO
    U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
    # first row
    U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) + (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1))
    U_t[n] += setelt(ll[1]) * setelt(rl[2]) * (C + (τ / 2) * braket(C, D) + (τ^2 / 6) * braket2(C, D))
    U_t[n] += setelt(ll[1]) * setelt(rl[3]) * (replaceprime(C' * C, 2, 1) + (τ / 3) * braket2(D, C))
    # second row
    U_t[n] += setelt(ll[2]) * setelt(rl[1]) * (τ * B + (τ^2 / 2) * braket(B, D) + (τ^3 / 6) * braket2(B, D))
    U_t[n] += setelt(ll[2]) * setelt(rl[2]) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) + (τ^2 / 6) * (braket(C, B, D) + braket2(A, D)))
    U_t[n] += setelt(ll[2]) * setelt(rl[3]) * (braket(A, C) + (τ / 3) * (braket(A, C, D) + braket2(B, C)))
    # third row
    U_t[n] += setelt(ll[3]) * setelt(rl[1]) * ((τ^2 / 2) * replaceprime(B' * B, 2, 1) + (τ^3 / 6) * braket2(D, B))
    U_t[n] += setelt(ll[3]) * setelt(rl[2]) * ((τ / 2) * braket(A, B) + (τ^2 / 6) * (braket(A, B, D) + braket2(C, B)))
    U_t[n] += setelt(ll[3]) * setelt(rl[3]) * (replaceprime(A' * A, 2, 1) + (τ / 3) * (braket(A, B, C) + braket2(D, A)))
  end

  # implementing OBC: project out upper row and fist column for right and left boundaries, respectively
  L = ITensor(linkindices[1])
  L[1] = 1.0

  R = ITensor(dag(linkindices[N+1]))
  R[1] = 1.0

  U_t[1] *= L
  U_t[N] *= R

  return U_t
end


function timeEvoMPO1_Ising(
  sites,
  AString::String,
  JA::Number,
  BString::String,
  JB::Number,
  CString::String,
  JC::Number,
  DString::String,
  JD::Number,
  t;
 )
  # measure τ in imaginary units for actualy time evolution
  τ = -1.0im * t
  N = length(sites)
  U_t = MPO(N)

  link_dimension = 2
  # startState = 3
  # endState = 1

  linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

  for n = 1:N
    # siteindex s
    s = sites[n]
    # left link index ll with daggered QN conserving direction (if applicable)
    ll = dag(linkindices[n])
    # right link index rl
    rl = linkindices[n+1]
    Id = op(sites, "Id", n)
    # A is possible exponential decay so test for "0"
    A = AString == "0" ? 0.0 * op(sites, "Id", n) : JA * op(sites, AString, n)
    B = JB * op(sites, BString, n)
    C = JC * op(sites, CString, n)
    D = JD * op(sites, DString, n)
    # Init ITensor inside MPO
    U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
    # first row
    U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) )
    U_t[n] += setelt(ll[1]) * setelt(rl[2]) * (C + (τ / 2) * braket(C, D))
    # second row
    U_t[n] += setelt(ll[2]) * setelt(rl[1]) * (τ * B + (τ^2 / 2) * braket(B, D))
    U_t[n] += setelt(ll[2]) * setelt(rl[2]) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) )
  end
  # location changes with convention of upper or lower triangular Matrix as MPO
  # here we have column-major convention i.e. lower-triangular
  L = ITensor(linkindices[1])
  L[1] = 1.0

  R = ITensor(dag(linkindices[N+1]))
  R[1] = 1.0

  U_t[1] *= L
  U_t[N] *= R

  return U_t
end







function buildExpHTFI(
  sites,
  λ::T1, # exponential decay base
  kinetic_coupling::T2, # interaction kinetic (tunneling) coupling
  ;
  kwargs...,
 )::ITensors.MPO where {T1<:Real} where {T2<:Real}

  # link_dimension
  d0 = dim(op(sites, "Id", 1), 1)
  link_dimension = 3
  
  startState = 3
  endState = 1


  hz = get(kwargs, :hz, 0.0)

  N = length(sites)

  hasqns(sites) ? error("The transverse field Ising model does not conserve total Spin Z") : true

  EType = eltype(union(λ, kinetic_coupling))

  # generate "regular" link indeces (i.e. without conserved QNs)
  linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

  H = MPO(sites)

  for n = 1:N
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
    H[n] += setelt(ll[startState]) * setelt(rl[2]) * op(sites, "Sx", n) * 2.0 # σˣ
    H[n] += setelt(ll[2]) * setelt(rl[2]) * op(sites, "Id", n) * λ  # λ Id,  on the diagonal
    H[n] += setelt(ll[2]) * setelt(rl[endState]) * op(sites, "Sx", n) * -2.0 * kinetic_coupling # Jxx σˣ
    if !iszero(hz)
      H[n] += setelt(ll[startState]) * setelt(rl[endState]) * op(sites, "Sz", n) * 2.0 * hz # hz σᶻ
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


function build_ising_H(sites, JXX, hz)

  os = OpSum()
  for j=1:length(sites)-1
    os += JXX,"X",j,"X",j+1
    os += hz, "Z",j
  end
  os += hz, "Z", N
  H = MPO(os,sites)


end


function tfimMPO(sites, h::Float64)
  # Input operator terms which define a Hamiltonian
  N = length(sites)
  os = OpSum()
  for j in 1:(N - 1)
    os += -1, "X", j, "X", j + 1
  end
  for j in 1:N
    os += h, "Z", j
  end
  # Convert these terms to an MPO tensor network
  return MPO(os, sites)
end



###############
############### USAGE EXAMPLE requires ITensors and ITensorTDVP packages:
###############

using ITensors, ITensorTDVP, Printf

euler = 2.7182818284590450908

N = 40      # System size
λ = 0.0 # 1/euler # base of exponential interaction 
JXX = 1.0   # spin x -- spin x coupling
hz = 1.0    # local magnetic field in z direction

dt = 0.05   # time step

SVD_cutoff = 1e-12    # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 300       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)
# initial state
psi_prod = productMPS(ComplexF64, sites, "↑")

# number of time steps
nSteps = 50


# function timeEvoMPO1_Ising takes in Spin operators while
# buildExpHTFI assumes internally Pauli matrices, so offset with factor of S⁻¹ = 2
#Ut1 = timeEvoMPO1_Ising(sites, "Id", λ, "Sx", 2.0, "Sx", -2.0 * JXX, "Sz", 2hz, dt)
Ut1 = timeEvoMPO1_Ising(sites, "Id", λ, "X", 1.0, "X", -1.0 * JXX, "Z", hz, dt)

psi_u1 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u1[:] = apply(Ut1, psi_u1; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u1))")
end


#Ut2 = timeEvoMPO2_Ising(sites, "Id", λ, "Sx", 2.0, "Sx", -2.0 * JXX, "Sz", 2hz, dt)
Ut2 = timeEvoMPO2_Ising(sites, "Id", λ, "X", 1.0, "X", -1.0 * JXX, "Z", hz, dt)

psi_u2 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u2[:] = apply(Ut2, psi_u2; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u2))")
end


Hexp = buildExpHTFI(sites, λ, JXX, hz=hz)

#psi_tdvp1 = deepcopy(psi_prod)

psi_tdvp1 = ITensorTDVP.tdvp(
           Hexp,
           psi_prod,
           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
           nsweeps = nSteps,
           maxdim = maxbondim,
           cutoff = SVD_cutoff,
           normalize = true,
           outputlevel=1,
)

Hisi = build_ising_H(sites, -1., 1)



#psi_tdvp2 = deepcopy(psi_prod)
psi_tdvp2 = ITensorTDVP.tdvp(
           Hisi,
           psi_prod,
           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
           nsweeps = nSteps,
           maxdim = maxbondim,
           cutoff = SVD_cutoff,
           normalize = true,
           outputlevel=1,
)

Hisi2 = tfimMPO(sites, 1.)

psi_tdvp3 = ITensorTDVP.tdvp(
           Hisi2,
           psi_prod,
           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
           nsweeps = nSteps,
           maxdim = maxbondim,
           cutoff = SVD_cutoff,
           normalize = true,
           outputlevel=1,
)

tdvp_u1 = abs(inner(psi_tdvp1, psi_u1))
tdvp_u2 = abs(inner(psi_tdvp1, psi_u2))

u1_u2 = abs(inner(psi_u1, psi_u2))

tdvp1_tdvp2 = abs(inner(psi_tdvp1, psi_tdvp2))

tdvp1_tdvp3 = abs(inner(psi_tdvp1, psi_tdvp3))


println("time:\tdt=$(dt),\tnSteps=$(nSteps),\tt_tot = $(dt*nSteps)")
println("<TDVP(ψ) | Ut1(ψ)> = $(tdvp_u1)")
println("<TDVP(ψ) | Ut2(ψ)> = $(tdvp_u2)")
println(" <Ut1(ψ) | Ut2(ψ)> = $(u1_u2)")
println("<TDVP1(ψ) | TDVP2(ψ)> = $(tdvp1_tdvp2)")
println("<TDVP1(ψ) | TDVP3(ψ)> = $(tdvp1_tdvp3)")