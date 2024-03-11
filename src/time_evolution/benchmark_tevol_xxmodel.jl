
using ITensors, ITensorTDVP
include("../power_method/utils.jl")
include("../models/ising.jl")
include("../models/xxzmodel.jl")
include("../power_method/compute_entropies.jl")


N = 20      # System size

# number of time steps
nSteps = 21


JXX = 1.0   # spin x -- spin x coupling
hz = 1.0

dt = 0.1  # time step

SVD_cutoff = 1e-12    # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 400       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
#psi_prod = productMPS(ComplexF64, sites, "↑")
psi_prod = randomMPS(sites, 1)


Ut1 = build_expH_XX_murg_from_ising(sites, JXX, hz, dt)

println("Apply Murg")
psi_u1 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u1[:] = apply(Ut1, psi_u1; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u1))")
end

# Ut_svd = build_expH_XX_svd(sites, JXX, dt)

# println("Apply SVD")
# psi_svd = deepcopy(psi_prod)
# @time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
#   psi_svd[:] = apply(Ut_svd, psi_svd; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
#   println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_svd))")
# end


Hisi = build_H_XX(sites, JXX, hz)

println("TDVP autoMPO H_XX")
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



tdvp_u1 = abs(inner(psi_tdvp1, psi_u1))



println("time:\tdt=$(dt),\tnSteps=$(nSteps),\tt_tot = $(dt*nSteps)")

println("<TDVP1(ψ) | U_murg(ψ)> = $(tdvp_u1)")

# tdvp_svd = abs(inner(psi_tdvp1, psi_svd))

# println("<TDVP1(ψ) | U_svd(ψ)> = $(tdvp_svd)")

ev_o1 = expect(psi_u1,"Sz")
ev_tdvp = expect(psi_tdvp1,"Sz")
# ev_svd = expect(psi_svd,"Sz")


println(ev_o1)
println(ev_tdvp)
# println(ev_svd)


println(vn_entanglement_entropy(psi_u1))
println(vn_entanglement_entropy(psi_tdvp1))
# println(vn_entanglement_entropy(psi_svd))



