using Test

using ITensors, ITensorMPS
using ITransverse 


@testset "Diagonalization of RTM using symmetric gauges" begin
s = siteinds("S=3/2", 20)

ll = random_mps(ComplexF64, s, linkdims=40)

eigs_l = diagonalize_rtm_left_gen_sym(ll, bring_left_gen=true)

eigs_r = diagonalize_rtm_right_gen_sym(ll, bring_right_gen=true)

for jj in eachindex(eigs_l)
    eL = eigs_l[jj]
    eR = eigs_r[end-jj+1]
    @test eL[abs.(eL) .> 1e-14] ≈ eR[abs.(eR) .> 1e-14]
end

end