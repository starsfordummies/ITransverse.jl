
using ITensors, ITensorTDVP
using ITransverse
using ITransverse: ChainModels.build_expH_XX_murg_from_ising, 
ChainModels.build_expH_XX_svd, ChainModels.build_expH_XX_SpSm_svd, ChainModels.build_H_XX

using Plots


N = 20     # System size

# number of time steps
nSteps = 16


JXX = 1.0   # spin x -- spin x coupling
hz = 0.0

dt = 0.05  # time step

SVD_cutoff = 1e-12    # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 400       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
#psi_prod = productMPS(ComplexF64, sites, "↑")
psi_prod = randomMPS(sites, 1)


Ut1 = build_expH_XX_murg_from_ising(sites, JXX, dt)

println("Apply Murg")
psi_u1 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u1[:] = apply(Ut1, psi_u1; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u1))")
end


Ut_svd = build_expH_XX_svd(sites, JXX, dt)

println("Apply SVD")
psi_svd = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_svd[:] = apply(Ut_svd, psi_svd; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_svd))")
end

Ut_svd = build_expH_XX_SpSm_svd(sites, JXX, dt)

println("Apply SVD PM")
psi_svd2 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_svd2[:] = apply(Ut_svd, psi_svd2; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_svd2))")
end

H_XX = build_H_XX(sites, JXX, hz)

println("TDVP autoMPO H_XX")
psi_tdvp1 = ITensorTDVP.tdvp(
          H_XX,
          psi_prod,
          -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
          nsweeps = nSteps,
          maxdim = maxbondim,
          cutoff = SVD_cutoff,
          normalize = true,
          outputlevel=1,
)



tdvp_u1 = abs(inner(psi_tdvp1, psi_u1))
tdvp_svd = abs(inner(psi_tdvp1, psi_svd))
tdvp_svd2 = abs(inner(psi_tdvp1, psi_svd2))
svd1_svd2 = abs(inner(psi_svd, psi_svd2))


println("time:\tdt=$(dt),\tnSteps=$(nSteps),\tt_tot = $(dt*nSteps)")

println("<TDVP1(ψ) | U_murg(ψ)> = $(tdvp_u1)")
println("<TDVP1(ψ) | svd(ψ)> = $(tdvp_svd)")
println("<TDVP1(ψ) | svd2(ψ)> = $(tdvp_svd2)")
println("<svd1(ψ) | svd2(ψ)> = $(svd1_svd2)")

# tdvp_svd = abs(inner(psi_tdvp1, psi_svd))

# println("<TDVP1(ψ) | U_svd(ψ)> = $(tdvp_svd)")

ev_o1 = expect(psi_u1,"Sz")

ev_tdvp = expect(psi_tdvp1,"Sz")
ev_svd = expect(psi_svd,"Sz")
ev_svd2 = expect(psi_svd2,"Sz")


pl1 = plot(ev_tdvp, label="tdvp")
scatter!(pl1,ev_o1, label="murg 2xising")
plot!(pl1,ev_svd, label= "SVD")
plot!(pl1,ev_svd2, label="S+S- SVD")

# println(ev_svd)


# println(vn_entanglement_entropy(psi_u1))
# println(vn_entanglement_entropy(psi_tdvp1))
# println(vn_entanglement_entropy(psi_svd))



