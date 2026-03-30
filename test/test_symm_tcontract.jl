using ITensors, ITensorMPS, ITransverse
using Test

ss = siteinds(2, 12)
psi = random_mps(ComplexF64, ss, linkdims=10)
oo = random_mpo(ss) + im*random_mpo(ss) + random_mpo(ss)

opsi = applyn(oo, psi;truncate=false)

tpsi1, sv1 = tapply(oo, psi; alg="naiveRTMsym", maxdim=30, cutoff=1e-30, method="SVD")
tpsi1L, sv1L  = tapply(oo, psi; alg="naiveRTMsym", maxdim=30, cutoff=1e-30, method="SVD", direction=:left)
tpsi2, sv2 = tapply(oo, psi; alg="naiveRTMsymRTM", maxdim=30, cutoff=1e-30)
tpsi3, sv3 = tapply(oo, psi; alg="RTMsym", maxdim=30, cutoff=1e-30)

fidelity(opsi, tpsi1)
fidelity(tpsi1, tpsi1L)
fidelity(tpsi1, tpsi2)
fidelity(tpsi1, tpsi3)

gen_fidelity(opsi,opsi)
gen_fidelity(tpsi1, tpsi1L)
gen_fidelity(tpsi1, tpsi2)
gen_fidelity(tpsi1, tpsi3)


tpsi1, sv1 = tapply(oo, psi; alg="naiveRTMsym", maxdim=30, cutoff=1e-8, method="SVD")
tpsi1L, sv1L  = tapply(oo, psi; alg="naiveRTMsym", maxdim=30, cutoff=1e-8, method="SVD", direction=:left)
tpsi2, sv2 = tapply(oo, psi; alg="naiveRTMsymRTM", maxdim=30, cutoff=1e-8)
tpsi3, sv3 = tapply(oo, psi; alg="RTMsym", maxdim=30, cutoff=1e-8)

maxlinkdim.([opsi, tpsi1, tpsi1L, tpsi2, tpsi3])

fidelity(opsi, tpsi1)
fidelity(tpsi1, tpsi1L)
fidelity(tpsi1, tpsi2)
fidelity(tpsi1, tpsi3)

gen_fidelity(opsi,opsi)
gen_fidelity(tpsi1, tpsi1L)
gen_fidelity(tpsi1, tpsi2)
gen_fidelity(tpsi1, tpsi3)
