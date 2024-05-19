
using ITensors, ITensorTDVP, Printf
include("../models/ising.jl")


N = 30      # System size

# number of time steps
nSteps = 35


JXX = 1.0   # spin x -- spin x coupling
hz = 0.8   # local magnetic field in z direction

dt = 0.1  # time step

SVD_cutoff = 1e-12    # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 300       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
#psi_prod = productMPS(ComplexF64, sites, "↑")
psi_prod = random_mps(sites,10)



Ut3 = build_expH_ising_murg_YY(sites, JXX, dt)

println("Apply murg")
psi_u3 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u3[:] = apply(Ut3, psi_u3; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u3))")
end




Hisi = build_H_ising_YY(sites, JXX, 0.)

println("TDVP autoMPO isi")
psi_tdvp1 = ITensorTDVP.tdvp(
          Hisi,
          psi_prod,
          -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
          nsweeps = nSteps,
          maxdim = maxbondim,
          cutoff = SVD_cutoff,
          normalize = true,
          outputlevel=1,
)




tdvp_u3 = abs(inner(psi_tdvp1, psi_u3))



println("<TDVP1(ψ) | U_murg(ψ)> = $(tdvp_u3)")
