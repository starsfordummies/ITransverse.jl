using Test
using ITensors
using ITransverse 
using ITransverse.ITenUtils

s = siteinds("S=1/2", 6)
s2 = sim(s)

p = random_mps(s)
p2 = random_mps(s2)

o = random_mpo(s)
o2 = random_mpo(s2)

match_siteinds!(p, p2)

@test siteinds(p) == siteinds(p2)

match_siteinds!(o, o2)

@test siteinds(o) == siteinds(o2)
