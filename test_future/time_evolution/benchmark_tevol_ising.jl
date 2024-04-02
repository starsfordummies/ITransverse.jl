using ITensors, ITensorTDVP
include("../myutils/pparams.jl")
include("../models/brakets.jl")
include("../models/ising.jl")


N = 60      # System size

# number of time steps
nSteps = 30


JXX = 1.0   # spin x -- spin x coupling
hz = 0.8   # local magnetic field in z direction

dt = 0.1  # time step

SVD_cutoff = 1e-4   # cutoff for singular vaulues smaller than SVD_cutoff
maxbondim = 300       # maximum bond dimension allowed

# define local degrees of freedom
sites = siteinds("S=1/2", N; conserve_qns = false)

# initial state
psi_prod = productMPS(ComplexF64, sites, "↑")



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
psi_u3 = deepcopy(psi_prod)
@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
  psi_u3[:] = apply(Ut3, psi_u3; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u3))")
end


#=
Ut4 = build_expH_ising_2o_Jan(sites, JXX, hz, dt)

psi_u4 = deepcopy(psi_prod)

#@show siteinds(psi_u2_j)
#@show siteinds(exphp_jan)

@time for (nt, t) in enumerate(range(dt, step = dt, length = nSteps))
    psi_u4[:] = apply(Ut4, psi_u4; normalize = true, cutoff = SVD_cutoff, maxdim = maxbondim)
    println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u4))")
end
=#
psi_u4 = randomMPS(sites)

Hisi = build_H_ising(sites, JXX, hz)

println("TDVP autoMPO isi")
psi_tdvp1 = tdvp(
          Hisi,
          psi_prod,
          -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
          nsweeps = nSteps,
          maxdim = maxbondim,
          cutoff = SVD_cutoff,
          normalize = true,
          outputlevel=1,
)



### These checks should be trivial 

# Hisi_man = build_H_ising_manual(sites, JXX, hz)


# println("TDVP manual isi")
# psi_tdvp2 = ITensorTDVP.tdvp(
#           Hisi_man,
#           psi_prod,
#           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
#           nsweeps = nSteps,
#           maxdim = maxbondim,
#           cutoff = SVD_cutoff,
#           normalize = true,
#           outputlevel=1,
# )


# Hisi_man_low = build_H_ising_manual_lowtri(sites, JXX, hz)


# println("TDVP manual isi")
# psi_tdvp3 = ITensorTDVP.tdvp(
#           Hisi_man_low,
#           psi_prod,
#           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
#           nsweeps = nSteps,
#           maxdim = maxbondim,
#           cutoff = SVD_cutoff,
#           normalize = true,
#           outputlevel=1,
# )


# tdvp1_tdvp2 = abs(inner(psi_tdvp1, psi_tdvp2))
# tdvp1_tdvp3 = abs(inner(psi_tdvp1, psi_tdvp3))
# tdvp2_tdvp3 = abs(inner(psi_tdvp2, psi_tdvp3))


# println('<TDVP1(ψ) | TDVP2(ψ)> = $(tdvp1_tdvp2)')
# println("<TDVP1(ψ) | TDVP3(ψ)> = $(tdvp1_tdvp3)")
# println("<TDVP2(ψ) | TDVP3(ψ)> = $(tdvp2_tdvp3)")




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

