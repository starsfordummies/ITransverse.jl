using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state


mp = IsingParams(1, 0.7, 0)


sites = siteinds("S=1/2", 20; conserve_qns=false)
Ut = build_expH_ising_murg(sites, mp, 0.1)
Ut2 = ITransverse.ChainModels.build_expH_ising_murg_new(sites, mp, 0.1)

Utvec, _ = ITransverse.ITenUtils.vectorize_mpo(Ut)
Ut2vec, _ = ITransverse.ITenUtils.vectorize_mpo(Ut2)
fidelity(Utvec, Ut2vec)

norm.(Utvec) ≈ norm.(Ut2vec)

psi0 = MPS(sites, "Up")


psiT = deepcopy(psi0)
for jj = 1:20
    psiT = apply(Ut2, psiT)
end

expect(psiT, "Z")[div(length(psiT)+1,2)]


ss = siteinds("S=1/2", 20; conserve_szparity=true)

psi0 = MPS(ss, "Up")

Ut_sym = ITransverse.ChainModels.build_expH_ising_murg_new(ss, mp, 0.1)

UUt_sym = ITransverse.folded_UUt(Ut_sym)

rho0 = outer(psi0,dag(psi0)')

rho0v, combis = ITransverse.ITenUtils.vectorize_mpo(rho0)
rhoT = deepcopy(rho0v)

rhoT = replace_siteinds(rhoT, firstsiteinds(UUt_sym))

for jj = 1:20
    rhoT = apply(UUt_sym, rhoT)
end

local_ops = [op("I", s) for s in siteinds(psi0)]
local_ops[div(length(rhoT)+1,2)] = op("Z", siteind(psi0, div(length(psi0)+1,2)))

o_local_ops = MPO(local_ops)

vo_local_ops, _ = ITransverse.ITenUtils.vectorize_mpo(o_local_ops)

inner(rhoT, vo_local_ops)