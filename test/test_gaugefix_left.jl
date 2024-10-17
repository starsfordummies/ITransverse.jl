using ITensors, ITensorMPS
using Test
using ITransverse.ITenUtils

s = siteinds("S=1/2", 30)
psi = random_mps(s, linkdims=50)

psi_l = gaugefix_left(psi)

for ii in eachindex(psi_l)[end:-1:2]
    #print(ii)
    env = psi_l[ii] * prime(dag(psi_l[ii]), linkind(psi_l, ii-1))
    @test isid(env)
end

env1 =  psi_l[1] * prime(dag(psi_l[1]), linkind(psi_l, 1))

@test isdiag(env1)