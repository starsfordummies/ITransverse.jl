using Test

using ITensors, ITensorMPS
using ITransverse 
using ITransverse: diagonalize_rtm_right_gen_sym

function nonzero_match(A, B; tol=1e-8)
    # Filter nonzeros
    A_nz = filter(x -> abs(x) > tol, A)
    B_nz = filter(x -> abs(x) > tol, B)
    
    # Count how many times each unique element appears (with tolerance)
    # For complex numbers, use approximate matching
    # Here we sort and compare
    sortA = sort(A_nz, by=abs)
    sortB = sort(B_nz, by=abs)
    
    return length(sortA) == length(sortB) &&
           all(isapprox.(sortA, sortB, atol=tol))
end

@testset "Diagonalization of RTM using symmetric gauges" begin
s = siteinds(4, 20)

ll = random_mps(ComplexF64, s, linkdims=40)

@info "Checking whether eigenvalues computed in left and right gauges match"
eigs_l = diagonalize_rtm_symmetric(ll; bring_left_gen=true)

# eigs_r sweeps from left to right, eigs_l the other way around
eigs_r = diagonalize_rtm_right_gen_sym(ll; bring_right_gen=true, normalize_factor=sqrt(overlap_noconj(ll,ll)))

# @show eigs_l[5]
# @show eigs_r[5]


@test nonzero_match(eigs_l[5], eigs_r[5]; tol=1e-9)
@test nonzero_match(eigs_l[10], eigs_r[10]; tol=1e-9)
@test nonzero_match(eigs_l[14], eigs_r[14]; tol=1e-9)
# @test eigs_l ≈ eigs_r 


end


@testset "Testing right gauge symmetric RTM diagonalization vs full diag" begin

    psi = random_mps(ComplexF64, siteinds("S=1/2",20), linkdims=30)
    
    mpslen = length(psi)
    
    psi_gauged = gen_canonical_right(psi)
    
    lenv= ITensors.OneITensor()
    s = siteinds(psi_gauged)
    
    for jj in 1:mpslen-1
    
        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )
    
        @assert ndims(lenv) == 2 
    
        vals = eigvals(lenv)
  
        if mpslen - jj < 5
            renv = ITensors.OneITensor()
            for kk in mpslen:-1:jj+1
                renv *= psi_gauged[kk]
                renv *= psi_gauged[kk]'
            end
    
            rtm_full = lenv * renv
            #@info "RTM full size: $(size(rtm_full))"
            vals_full, _ = eigen(rtm_full, inds(rtm_full,plev=0), inds(rtm_full,plev=1))
    
            @show jj 
            @show vals 
            @test nonzero_match(vals, array(vals_full))
        end
    
    end
    
    # @show (eigs_rho)
    # @show diag.(matrix.(eigs_rho_check))

    # @test diag.(matrix.((eigs_rho[end-3:end]))) ≈ diag.(matrix.(eigs_rho_check))
    
    end



@testset "Test whether computed Tr(τ_t^2) match " begin

    s = siteinds(4, 20)
    
    psi = random_mps(ComplexF64, s, linkdims=40)
    psic = deepcopy(psi)

    eigs_r = diagonalize_rtm_right_gen_sym(psi; bring_right_gen=true)
    eigs_l = diagonalize_rtm_symmetric(psi; bring_left_gen=true)

    all_ents = renyi_entropies(eigs_r, which_ents=[2.0])
    
    psin = psi/sqrt(overlap_noconj(psi,psi))
    r2_cut = log.(rtm2_contracted(psin, psin))/(1-2.)
    
    @test psi[7] == psic[7]

    @test all_ents["S2.0"] ≈ r2_cut
    
end
    

@testset "Symmetric renyi2 " begin

    s = siteinds(4, 20)
    psi = random_mps(ComplexF64, s, linkdims=40)

    r2_contract = gen_renyi2(psi, psi)
    r2sym = gensym_renyi_entropies(psi)["S2.0"]

    @test isapprox(r2_contract, r2sym)
end