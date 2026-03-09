using ITensors, ITensorMPS, ITransverse 
using Plots
using Test

@testset "XXZ Hamiltonians" begin
    JXY = 1.0
    ΔZZ = 0.8
    hz = 0.5

    ss = siteinds("S=1/2", 7)

    h1 = ITransverse.H_XXZ(ss, JXY::Real, ΔZZ::Real, hz::Real)
    h2 = ITransverse.H_XXZ_SpSm(ss, JXY::Real, ΔZZ::Real, hz::Real)

    @test h1 ≈ h2

end

# @testset "XXZ H vs exp(H) " begin
let
cutoff = 1e-10
maxdim = 128

ss = siteinds("S=1/2", 20)

N = length(ss)

neel = [isodd(i) ? "Up" : "Dn" for i in 1:N]

psi0 = MPS(ss, neel)

U_t = MPO(N)

JXX = 1.0
dt = 0.1 
Δ = 0.8

NT = 30

Hxxz = ITransverse.H_XXZ(ss, JXX, Δ, 0)
maxlinkdim(Hxxz)

eHXXZ_2o = ITransverse.expH_XXZ_2o(ss, JXX, Δ, 0; dt) 
eHXXZ_svd = ITransverse.expH_XXZ_svd(ss, JXX, Δ; dt) 

psi_tebd_2o = copy(psi0)
psi_tebd_svd = copy(psi0)

vals_tebd_2o = []
vals_tebd_svd = []

for kk = 1:NT
    psi_tebd_2o = apply(eHXXZ_2o, psi_tebd_2o; cutoff, maxdim)
    psi_tebd_svd = apply(eHXXZ_svd, psi_tebd_svd; cutoff, maxdim)

    @show maxlinkdim(psi_tebd_2o), expect(psi_tebd_2o, "Z")[div(length(ss),2)]
    @show maxlinkdim(psi_tebd_svd), expect(psi_tebd_svd, "Z")[div(length(ss),2)]

    push!(vals_tebd_2o,  expect(psi_tebd_2o, "Z")[div(length(ss),2)])
    push!(vals_tebd_svd,  expect(psi_tebd_svd, "Z")[div(length(ss),2)])

end


psi_tdvp = copy(psi0)

vals_tdvp = []
for kk = 1:NT
 psi_tdvp = tdvp(
        Hxxz,
        -dt*im,
        psi_tdvp;
        maxdim,
        cutoff,
        normalize=true,
        outputlevel=0,
    )
    @show maxlinkdim(psi_tdvp), expect(psi_tdvp, "Z")[div(length(ss),2)]
    push!(vals_tdvp,  expect(psi_tdvp, "Z")[div(length(ss),2)])
end


@test norm(vals_tdvp - vals_tebd_2o)/norm(vals_tdvp) < 5e-2
@test norm(vals_tebd_2o - vals_tebd_2o)/norm(vals_tdvp) < 5e-2

plot(vals_tdvp)
scatter!(vals_tebd_2o, label="2o")
scatter!(vals_tebd_svd, label="svd")

end




# Commutation of gates 


ss = siteinds("S=1/2", 20)

nn = 4
Xi = op(ss, "X", nn)
Yi = op(ss, "Y", nn)
Zi = op(ss, "Z", nn)

Xj = op(ss, "X", nn+1)
Yj = op(ss, "Y", nn+1)
Zj = op(ss, "Z", nn+1)

e1 = exp(im*0.1*(Xi * Xj + Yi * Yj + Δ * Zi * Zj))

nn = 5
Xi = op(ss, "X", nn)
Yi = op(ss, "Y", nn)
Zi = op(ss, "Z", nn)

Xj = op(ss, "X", nn+1)
Yj = op(ss, "Y", nn+1)
Zj = op(ss, "Z", nn+1)

e2 = exp(im*dt*(Xi * Xj + Yi * Yj + Δ * Zi * Zj))

cs = combiner(ss[nn],ss[nn+1], tags="csite")
e2c = e2 * cs * cs'

ITransverse.ITenUtils.check_symmetry_swap(e2c, inds(e2c)...)

e12 = replaceprime(contract(e1, replaceinds(e2, commoninds(e1,e2) => commoninds(e1,e2)' )), 2 => 1)
e21 =  replaceprime(contract(e2, replaceinds(e1, commoninds(e1,e2) => commoninds(e1,e2)' )), 2 => 1)

norm(e12 - e21)