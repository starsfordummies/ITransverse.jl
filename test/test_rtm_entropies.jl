using Test

using ITensors, ITensorMPS
using ITransverse 

function trim_near_zero(v, cutoff)
    mags = abs.(v)  # Elementwise magnitude, works for real and complex
    first_nz = findfirst(x -> x > cutoff, mags)
    last_nz = findlast(x -> x > cutoff, mags)
    if isnothing(first_nz) || isnothing(last_nz)
        return []
    end
    return v[first_nz:last_nz]
end

function equal_up_to_cutoff(a, b; cutoff=1e-8, rtol=1e-8)
    ta = trim_near_zero(a, cutoff)
    tb = trim_near_zero(b, cutoff)
    length(ta) == length(tb) && isapprox(ta, tb; atol=cutoff, rtol=rtol)
end


@testset "Diagonalization of RTM using symmetric gauges" begin
s = siteinds(4, 20)

ll = random_mps(ComplexF64, s, linkdims=40)
normalize_factor = rand()*10.

@info "Checking whether eigenvalues computed in left and right gauges match"
eigs_l = diagonalize_rtm_left_gen_sym(ll; bring_left_gen=true, normalize_factor)

# eigs_r sweeps from left to right, eigs_l the other way around
eigs_r = diagonalize_rtm_right_gen_sym(ll; bring_right_gen=true, normalize_factor)

# @info eigs_l
# @info eigs_r
@test equal_up_to_cutoff(eigs_l[5], eigs_r[5]; cutoff=1e-12, rtol=1e-12)
@test equal_up_to_cutoff(eigs_l[10], eigs_r[10]; cutoff=1e-12, rtol=1e-12)
@test equal_up_to_cutoff(eigs_l[15], eigs_r[15]; cutoff=1e-12, rtol=1e-12)
# @test eigs_l ≈ eigs_r 


end


@testset "Testing right gauge symmetric RTM diagonalization vs full diag" begin

    psi = quick_mps(20, 30)
    
    mpslen = length(psi)
    
    psi_gauged = gen_canonical_right(psi)
    
    lenv= ITensor(1.)
    s = siteinds(psi_gauged)
    
    eigs_rho = []
    eigs_rho_check = []
    
    for jj in 1:mpslen-1
    
        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )
    
        @assert ndims(lenv) == 2 
    
        vals, vecs = eigen(lenv, ind(lenv,1), ind(lenv,2))
    
        push!(eigs_rho, vals)
    
        if mpslen - jj < 5
            renv = ITensor(1.)
            for kk in mpslen:-1:jj+1
                renv *= psi_gauged[kk]
                renv *= psi_gauged[kk]'
            end
    
            rtm_full = lenv * renv
            #@info "RTM full size: $(size(rtm_full))"
            vals_full, _ = eigen(rtm_full, inds(rtm_full,plev=0), inds(rtm_full,plev=1))
    
            push!(eigs_rho_check, vals_full)
        end
    
    end
    
    
    @test diag.(matrix.((eigs_rho[end-3:end]))) ≈ diag.(matrix.(eigs_rho_check))
    
    end



@testset "Test whether computed Tr(τ_t^2) match " begin

    s = siteinds(4, 20)
    
    psi = random_mps(ComplexF64, s, linkdims=40)
    psic = deepcopy(psi)

    eigs_r = diagonalize_rtm_right_gen_sym(psi; bring_right_gen=true)
    eigs_l = diagonalize_rtm_left_gen_sym(psi; bring_left_gen=true)

    all_ents = build_entropies(eigs_r, [2.0])
    
    r2_cut = rtm2_contracted(psi, psi)
    
    @test psi[7] == psic[7]

    @test all_ents["S2.0"] ≈ r2_cut
    
    end
    