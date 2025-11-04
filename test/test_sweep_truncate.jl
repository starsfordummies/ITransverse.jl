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

    psi_trunc, phi_trunc, s = truncate_lsweep(psi, phi, cutoff=cut_sv, chi_max=chimaxs)

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
        
    cut_sv = 1e-14

    for chimaxs = [256, 32]
        sites = siteinds("S=1/2", 60)

        a = random_mps(ComplexF64, sites, linkdims=chimaxs);
        b = random_mps(ComplexF64, sites, linkdims=chimaxs);
        
        @info "Cutting at $(cut_sv)"
        @info overlap_noconj(a,b)

        atr, btr, s1 = truncate_rsweep(a,b, cutoff=cut_sv, chi_max=chimaxs);
        atr2, btr2, s2 = truncate_rsweep(a,b, cutoff=cut_sv, chi_max=chimaxs, fast=true);
        s3 = truncate_rsweep!(a,b, cutoff=cut_sv, chi_max=chimaxs)
        
        @test norm(atr-atr2) < 1e-14
        @test norm(btr-btr2) < 1e-14
        @test norm(a-atr2) < 1e-14
        @test norm(b-btr2) < 1e-14
        @test s1 ≈ s2 
        # @test s1 ≈ s3   # truncate_rsweep!() does not return entropies anymore for speed
        
        @show fidelity(atr,atr2)
        @show fidelity(btr,btr2)

        @show fidelity(atr,a)
        @show fidelity(atr2,a)

        @show fidelity(btr,b)
        @show fidelity(btr2,b)

        @show overlap_noconj(a,b)
        @show overlap_noconj(atr,btr)
        @show overlap_noconj(atr2,btr2)
    end
end



@testset "Test that truncate_rsweep_sym(a, fast=true) gives same as truncate_rsweep(a) " begin
        
    for chimaxs = [256, 32]

        cut_sv = 1e-14
        sites = siteinds("S=1/2", 60)

        a = random_mps(ComplexF64, sites, linkdims=chimaxs);
        
        @info "Cutting at $(cut_sv)"
        @show overlap_noconj(a,a)

        atr1,s1 = truncate_rsweep_sym(a, cutoff=cut_sv, chi_max=chimaxs, method="SVD");
        atr2,s2 = truncate_rsweep_sym(a, cutoff=cut_sv, chi_max=chimaxs, method="SVD", fast=true);
        
        @test norm(atr1-atr2) < 1e-14
        @test s1 ≈ s2 
        # @test s1 ≈ s3   # truncate_rsweep!() does not return entropies anymore for speed
        
        @info fidelity(atr1,atr2)


        @info overlap_noconj(a,atr1)
        @info overlap_noconj(a,atr2)
    end
end