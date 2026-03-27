using ITensors, ITensorMPS 
using Test

ITensors.set_warn_order(22) 

@testset "check that eigen(tauA) = eigen(tauB)  (but svd(tauA) != svd(tauB))" begin
NN = 13

ss = siteinds("S=1/2", NN)

psi = random_mps(ComplexF64, ss, linkdims = 64)
phi = random_mps(ComplexF64, ss, linkdims = 48)


psip = prime(siteinds, psi)
phip = prime(siteinds, phi, 2)


for cut = [5, 9] 

left = ITensor(1)

for jj = 1:cut
    left *= psi[jj]
    left *= phi[jj]
end

for jj = cut+1:NN
    left *= psip[jj]
    left *= phip[jj]
end

left *= combiner(inds(left,plev=1), tags="left")
left *= combiner(inds(left,plev=2), tags="right")

eigs_rho_left, _ = eigen(left, inds(left)...; cutoff=1e-14)
_, svs_rho_left, _ = svd(left, ind(left,1); cutoff=1e-14)

right = ITensor(1)

for jj = reverse(cut+1:NN)
    right *= psi[jj]
    right *= phi[jj]
end

for jj = reverse(1:cut)
    right *= psip[jj]
    right *= phip[jj]
end


right *= combiner(inds(right,plev=1), tags="left")
right *= combiner(inds(right,plev=2), tags="right")


eigs_rho_right, _ = eigen(right, inds(right)...; cutoff=1e-14)
_, svs_rho_right, _ = svd(right, ind(right,1); cutoff=1e-14)


neigs = min(dim(eigs_rho_left,1), dim(eigs_rho_right,1))


@test diag(eigs_rho_left)[1:neigs] ≈ diag(eigs_rho_right)[1:neigs]

@test norm(diag(eigs_rho_left)[neigs+1:end]) < 1e-14
@test norm(diag(eigs_rho_right)[neigs+1:end]) < 1e-14

@test norm(diag(svs_rho_left)[1:4] - diag(svs_rho_right)[1:4]) > 0.02

end

end

ITensors.reset_warn_order()
