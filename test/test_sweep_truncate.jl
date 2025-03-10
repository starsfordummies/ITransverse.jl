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
    psi = normalize(add(psi, psi, maxdim=chimaxs, mindim=chimaxs))
    phi = normalize(add(dag(psi), psi, maxdim=chimaxs, mindim=chimaxs))

    @test linkdims(psi) == linkdims(phi)

    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = deepcopy(psi)
    phi_c = deepcopy(phi)

    psi_trunc, phi_trunc, s, ov = truncate_lsweep(psi, phi, cutoff=cut_sv, chi_max=chimaxs)

    # test we don't mess up with data
    @test inner(psi, psi_c) ≈ 1
    @test inner(phi, phi_c) ≈ 1


end

@testset "Right sweeps checks" begin

    psi = random_mps(ComplexF64, sites, linkdims=chimaxs)

    # this is a hacky way to give an overlap ~ 1/sqrt(2)
    psi = normalize(add(psi, psi, maxdim=chimaxs, mindim=chimaxs))
    phi = normalize(add(dag(psi), psi, maxdim=chimaxs, mindim=chimaxs))
    
    @test linkdims(psi) == linkdims(phi)
    
    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = deepcopy(psi)
    phi_c = deepcopy(phi)

    psi_trunc, phi_trunc, s = truncate_rsweep(psi, phi, cutoff=cut_sv, chi_max=chimaxs)
    
    # test we don't mess up with data
    @test inner(psi, psi_c) ≈ 1
    @test inner(phi, phi_c) ≈ 1
    
    @show linkdims(psi)
    @show linkdims(psi_trunc)

    @show norm(psi)
    @show norm(psi_trunc)
    @show inner(psi, psi_trunc)
    @show inner(phi, phi_trunc)
    
    
end


@testset "Silly test that truncate_sweep(a,b) gives same as truncate_sweep(b,a)" begin

a = random_mps(ComplexF64, sites, linkdims=chimaxs)
b = random_mps(ComplexF64, sites, linkdims=chimaxs)

atr, btr, ss = truncate_rsweep(a,b, cutoff=cut_sv, chi_max=chimaxs)
btr2, atr2, ss2 = truncate_rsweep(b,a, cutoff=cut_sv, chi_max=chimaxs)

@test atr ≈ atr2 
@test btr ≈ btr2 
@test ss ≈ ss2 

end


@testset "Test that truncate_rsweep(a,b, fast=true) gives same as truncate_rsweep(a,b) and  truncate_rsweep!" begin
        
    for cut_sv = [1e-14, 1e-8, 1e-4]

        chimaxs = 256
        sites = siteinds("S=1/2", 120)

        a = random_mps(ComplexF64, sites, linkdims=chimaxs);
        b = random_mps(ComplexF64, sites, linkdims=chimaxs);
        
        @info "Cutting at $(cut_sv)"
        @info overlap_noconj(a,b)

        atr, btr, s1 = truncate_rsweep(a,b, cutoff=cut_sv, chi_max=chimaxs);
        atr2, btr2, s2 = truncate_rsweep(a,b, cutoff=cut_sv, chi_max=chimaxs, fast=true);
        s3 = truncate_rsweep!(a,b, cutoff=cut_sv, chi_max=chimaxs)
        @info "cut little"
        @test norm(atr-atr2) < 1e-14
        @test norm(btr-btr2) < 1e-14
        @test norm(a-atr2) < 1e-14
        @test norm(b-btr2) < 1e-14
        @test s1 ≈ s2 
        @test s1 ≈ s3
        
        @info inner(atr,atr2)/norm(atr)/norm(atr2)
        @info inner(btr,btr2)/norm(btr)/norm(btr2)

        @info inner(atr,a)/norm(atr)/norm(a)
        @info inner(atr2,a)/norm(atr2)/norm(a)

        @info inner(btr,b)/norm(btr)/norm(b)
        @info inner(btr2,b)/norm(btr2)/norm(b)

        @info overlap_noconj(a,b)
        @info overlap_noconj(atr,btr)
        @info overlap_noconj(atr2,btr2)
    end
end