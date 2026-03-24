using ITensors, ITensorMPS
using ITransverse
using Test


""" Ideally here we'd like to truncate a left and a right MPS
in order to optimize their overlap. How close are the resulting two 
with respect to the original ones?
"""

cutoff=1e-12 
maxdim=100
mindim=100

sites = siteinds("S=1/2", 60)

@testset "Left sweeps checks" begin

    psi = random_mps(ComplexF64, sites, linkdims=maxdim)

    # this is a hacky way to give an overlap ~ 1/sqrt(2)
    psi = normalize(add(psi, psi; maxdim, mindim))
    phi = normalize(add(dag(psi); psi, maxdim, mindim))

    @test linkdims(psi) == linkdims(phi)

    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = copy(psi)
    phi_c = copy(phi)

    psi_trunc, phi_trunc, s = truncate_sweep(psi, phi; cutoff, maxdim, direction=:left)

    # test we don't mess up with data
    @test inner(psi, psi_c) ≈ 1
    @test inner(phi, phi_c) ≈ 1


end

@testset "Right sweeps checks" begin

    psi = random_mps(ComplexF64, sites, linkdims=maxdim)

    # this is a hacky way to give an overlap ~ 1/sqrt(2)
    psi = normalize(add(psi, psi; maxdim, mindim))
    phi = normalize(add(dag(psi), psi; maxdim, mindim))
    
    @test linkdims(psi) == linkdims(phi)
    
    @show overlap_noconj(psi,phi) # Should this be the maximum value (?)

    psi_c = copy(psi)
    phi_c = copy(phi)

    psi_trunc, phi_trunc, s = truncate_sweep(psi, phi; cutoff, maxdim, direction=:right)
    
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

a = random_mps(ComplexF64, sites, linkdims=maxdim)
b = random_mps(ComplexF64, sites, linkdims=maxdim)

atr, btr, ss = truncate_sweep(a,b; cutoff, maxdim, direction=:right)
btr2, atr2, ss2 = truncate_sweep(b,a; cutoff, maxdim, direction=:right)

@test atr ≈ atr2 
@test btr ≈ btr2 
@test ss ≈ ss2 

end
