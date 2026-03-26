using ITensors, ITensorMPS

@testset "RTM in ortho gauge " begin
N = 50
ss = siteinds("S=1/2", N)

# has ortho center = 1 
psi = random_mps(ss, linkdims=128)
phi = random_mps(ss, linkdims=128)

cut = N-4

tau_end = ITensor(1)
for jj = 1:cut
    tau_end *= psi[jj]
    tau_end *= phi[jj]
end


_, S, _ = svd(tau_end, linkind(psi, cut))

for jj = cut+1:length(ss)
    tau_end *= psi[jj]
    tau_end *= prime(siteinds,phi)[jj]
end

_, S2, _ = svd(tau_end, inds(tau_end,plev=0))

@test storage(S/sum(S)) ≈ storage(S2/sum(S2))

end 