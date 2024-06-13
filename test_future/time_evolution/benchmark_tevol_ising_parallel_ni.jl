using ITensors, ITensorMPS

using ITransverse
using ITransverse: build_expH_ising_parallel_field_murg
#using .IGensors

N = 40      # System size

# number of time steps
nSteps = 20


JXX = 1.0   # spin x -- spin x coupling
hz = 1.05   # local magnetic field in z direction
gx = 0

dt = 0.1  # time step

SVD_cutoff = 1e-10   # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 300       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
#psi_prod = productMPS(ComplexF64, sites, "↑")
psi_prod = productMPS(ComplexF64, sites, "+")


ev_tebdO1 = [] 
ev_tebdO2 = []
ev_tebdmurg = []
ev_tdvp = []

Ut = build_expH_ising_parallel_field_murg(sites, JXX, hz, dt)

println("Apply murg")
psi_u3 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u3[:] = apply(Ut3, psi_u3; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  push!(ev_tebdmurg, expect(psi_u3, "Z", sites=round(Int,N/2)))

  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u3))")
end

psi_u4 = random_mps(sites)

Hisipar = build_H_ising_parallel_field(sites, JXX, hz)

println("TDVP autoMPO isi")
psi_tdvp1 = tdvp(
          Hisipar,
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
tdvp_u3 = abs(inner(psi_tdvp1, psi_u3))
tdvp_u4 = abs(inner(psi_tdvp1, psi_u4))



u1_u2 = abs(inner(psi_u1, psi_u2))
u1_u3 = abs(inner(psi_u1, psi_u3))
u3_u4 = abs(inner(psi_u3, psi_u4))


println("time:\tdt=$(dt),\tnSteps=$(nSteps),\tt_tot = $(dt*nSteps)")
println("<TDVP1(ψ) | U_o1(ψ)> = $(tdvp_u1)")
println("<TDVP1(ψ) | U_o2(ψ)> = $(tdvp_u2)")
println("<TDVP1(ψ) | U_murg(ψ)> = $(tdvp_u3)")
println("<TDVP1(ψ) | U_o2j(ψ)> = $(tdvp_u4)")


println(" <U_o1(ψ) | U_o2(ψ)> = $(u1_u2)")
println(" <U_o1(ψ) | Umurg(ψ)> = $(u1_u3)")
println(" <Umurg(ψ) | U_oj2(ψ)> = $(u3_u4)")

