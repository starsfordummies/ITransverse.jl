
using ITensors, ITensorTDVP
using Plots

using ITransverse: build_expH_ising_1o, build_expH_ising_2o, build_expH_ising_murg, build_H_ising

N = 40      # System size


JXX = 1.0   # spin x -- spin x coupling
hz = 0.4   # local magnetic field in z direction

dt = 0.1  # time step

SVD_cutoff = 1e-14    # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 300       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
psi_prod = productMPS(ComplexF64, sites, "↑")
#psi_prod = productMPS(ComplexF64, sites, "+")


nSteps = 30



Ut1 = build_expH_ising_1o(sites, JXX, hz, dt)

println("Apply O1")
psi_u1 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u1[:] = apply(Ut1, psi_u1; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u1))")
end


Ut2 = build_expH_ising_2o(sites, JXX, hz, dt)

println("Apply O2")
psi_u2 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u2[:] = apply(Ut2, psi_u2; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u2))")
end

Ut3 = build_expH_ising_murg(sites, JXX, hz, dt)

println("Apply murg")
evs_x_midchain = []
evs_z_midchain = []
psi_u3 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u3[:] = apply(Ut3, psi_u3; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u3))")
  push!(evs_x_midchain, expect(psi_u3, "X")[trunc(Int,(length(psi_u3)/2))])
  push!(evs_z_midchain, expect(psi_u3, "Z")[trunc(Int,(length(psi_u3)/2))])

end




Hisi = build_H_ising(sites, JXX, hz)

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

# Expectation values after 
ev_o1 = expect(psi_u1,"Z")
ev_o2 = expect(psi_u2,"Z")
ev_o3 = expect(psi_u3,"Z")
ev_td = expect(psi_tdvp1,"Z")

# plotlyjs()

pl1 = plot(ev_o1)
plot!(pl1, ev_o2)
plot!(pl1, ev_o3)
plot!(pl1, ev_td, label="tdvp")

pl2 = scatter(evs_x_midchain, label="X")
scatter!(pl2, evs_z_midchain, label="Z")
scatter!(pl2,[nSteps], [expect(psi_tdvp1,"Z", sites=div(N,2))],markersize=6)

plot(pl1, pl2)

