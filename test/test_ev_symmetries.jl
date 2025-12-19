using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state


@testset "Test computing exp. values using symmetries, folded and unfolded" begin 



Lx = 20
Nt = 12

mp = IsingParams(1, 0.7, 0)


sites = siteinds("S=1/2", Lx; conserve_qns=false)
Ut =  build_Ut(sites, expH_ising_murg,  mp; dt=0.1)
Ut2 = build_Ut(sites, expH_ising_murg,  mp; dt=0.1)

Utvec, _ = ITransverse.ITenUtils.vectorize_mpo(Ut)
Ut2vec, _ = ITransverse.ITenUtils.vectorize_mpo(Ut2)

@test fidelity(Utvec, Ut2vec) ≈ 1

@test norm.(Utvec) ≈ norm.(Ut2vec)

psi0 = MPS(sites, "Up")


psiT = deepcopy(psi0)
for jj = 1:Nt
    psiT = apply(Ut2, psiT)
end



expect(psiT, "Z")[div(length(psiT)+1,2)]


ss = siteinds("S=1/2", Lx; conserve_szparity=true)

psi0 = MPS(ss, "Up")

Ut_sym = build_Ut(ss, expH_ising_murg, mp; dt=0.1)

UUt_sym = ITransverse.folded_UUt(Ut_sym)

rho0 = outer(psi0,dag(psi0)')

rho0v, combis = ITransverse.ITenUtils.vectorize_mpo(rho0)
rhoT = deepcopy(rho0v)

rhoT = replace_siteinds(rhoT, firstsiteinds(UUt_sym))

for jj = 1:Nt
    rhoT = apply(UUt_sym, rhoT, cutoff=1e-12)
end

local_ops = [op("I", s) for s in siteinds(psi0)]
local_ops[div(length(rhoT)+1,2)] = op("Z", siteind(psi0, div(length(psi0)+1,2)))

o_local_ops = MPO(local_ops)

vo_local_ops, _ = ITransverse.ITenUtils.vectorize_mpo(o_local_ops)

vo_local_ops = replace_siteinds(vo_local_ops, siteinds(rhoT))

siteinds(rhoT)[1]
siteinds(vo_local_ops)[1]

@test isapprox(expect(psiT, "Z")[div(length(psiT)+1,2)] , inner(rhoT, vo_local_ops); rtol=1e-4)


Nt = 12

mp = IsingParams(1, 0.7, 0)


ss3 = siteinds("S=1/2", 3; conserve_szparity=true)


psi0 = MPS(ss3, "Up")
rho0 = outer(psi0,dag(psi0)')
rho0v, combis = ITransverse.ITenUtils.vectorize_mpo(rho0)


Ut_sym = build_Ut(ss3, expH_ising_murg, mp; dt=0.1)

UUt_sym = ITransverse.folded_UUt(Ut_sym)
UUt_sym2 = ITransverse.folded_UUt(Ut_sym)

local_ops = [op("I", s) for s in siteinds(psi0)]
#local_ops[div(length(rhoT)+1,2)] = op("Z", siteind(psi0, div(length(psi0)+1,2)))

o_local_ops = MPO(local_ops)
o_local_ops = MPO(siteinds(psi0), "Id")
vo_local_ops, _ = ITransverse.ITenUtils.vectorize_mpo(o_local_ops)


rho0v = replace_siteinds(rho0v, firstsiteinds(UUt_sym))
vo_local_ops = replace_siteinds(vo_local_ops, siteinds(rho0v))


@test inner(vo_local_ops, apply(UUt_sym, replace_siteinds(rho0v, firstsiteinds(UUt_sym)))) ≈ 1 



psiL, Tc, psiR = ITransverse.construct_tMPS_tMPO(rho0v, fill(UUt_sym, Nt), vo_local_ops)

@test overlap_noconj(psiL, applyn(Tc, psiR)) ≈ 1


# Now the local operator 
local_ops_Z = [op("Z", s) for s in siteinds(psi0)]
o_local_ops_Z = MPO(local_ops_Z)
vo_local_ops_Z, _ = ITransverse.ITenUtils.vectorize_mpo(o_local_ops_Z)

_, Tc_Z, _ = ITransverse.construct_tMPS_tMPO(rho0v, fill(UUt_sym, Nt), vo_local_ops_Z)

lleft = psiL
rright = psiR

for jj = 1:div(Lx,2)
    lleft = applyns(Tc, lleft; cutoff=1e-12)
    rright = applyn(Tc, rright; cutoff=1e-12)
end

rright = replace_siteinds(rright, firstsiteinds(Tc_Z))
lleft = replace_siteinds(lleft, firstsiteinds(Tc_Z))

expval_LR(lleft, Tc_Z, rright)/overlap_noconj(lleft,rright)

@test isapprox(expect(psiT, "Z")[div(length(psiT)+1,2)] , expval_LR(lleft, Tc_Z, rright)/overlap_noconj(lleft,rright); rtol=1e-5)

end