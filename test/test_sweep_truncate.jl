using Revise
using ITensors, ITensorMPS
using ITransverse
using Test


""" Ideally here we'd like to truncate a left and a right MPS
in order to optimize their overlap. How close are the resulting two 
with respect to the original ones?
"""

chimaxs = 100
sites = siteinds("S=1/2", 60)
cut_sv = 1e-12

@testset "Left sweeps checks" begin

    psi = random_mps(ComplexF64, sites, linkdims=chimaxs)

    # this is a hacky way to give an overlap ~ 1/sqrt(2)
    psi = normalize(sum(psi, psi, maxdim=chimaxs, mindim=chimaxs))
    phi = normalize(sum(dag(psi), psi, maxdim=chimaxs, mindim=chimaxs))

    @test linkdims(psi) == linkdims(phi)

    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = deepcopy(psi)
    phi_c = deepcopy(phi)

    psi_trunc, phi_trunc, s, ov = truncate_lsweep(psi, phi, cutoff=cut_sv, chi_max=chimaxs)

    # test we don't mess up with data
    @test inner(psi, psi_c) ≈ 1
    @test inner(phi, phi_c) ≈ 1


    @show linkdims(psi)
    @show linkdims(psi_trunc)
    @show norm(psiL)
    @show norm(l)
    @show inner(psi, psi_trunc)
    @show inner(phi, phi_trunc)

    @show overlap_noconj(l,r)/norm(l)/norm(r)

end

@testset "Right sweeps checks" begin

    psi = random_mps(ComplexF64, sites, linkdims=chimaxs)

    # this is a hacky way to give an overlap ~ 1/sqrt(2)
    psi = normalize(sum(psi, psi, maxdim=chimaxs, mindim=chimaxs))
    phi = normalize(sum(dag(psi), psi, maxdim=chimaxs, mindim=chimaxs))
    
    @test linkdims(psi) == linkdims(phi)
    
    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = deepcopy(psi)
    phi_c = deepcopy(phi)

    psi_trunc, phi_trunc, s, ov = truncate_rsweep(psi, phi, cutoff=cut_sv, chi_max=chimaxs)
    
    # test we don't mess up with data
    @test inner(psi, psi_c) ≈ 1
    @test inner(phi, phi_c) ≈ 1
    
    @show linkdims(psi)
    @show linkdims(psi_trunc)

    @show norm(psiL)
    @show norm(l)
    @show inner(psi, psi_trunc)
    @show inner(phi, phi_trunc)
    
    @show overlap_noconj(l,r)/norm(l)/norm(r)
    
end
