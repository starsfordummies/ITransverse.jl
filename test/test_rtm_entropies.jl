using Test

using ITensors, ITensorMPS
using ITransverse 


@testset "Diagonalization of RTM using symmetric gauges" begin
s = siteinds(4, 20)

ll = random_mps(ComplexF64, s, linkdims=40)
normalize_factor = rand()*10.

@info "Checking whether results in left and right gauges match"
eigs_l = diagonalize_rtm_left_gen_sym(ll; bring_left_gen=true, normalize_factor)

# eigs_r sweeps from left to right, eigs_l the other way around
eigs_r = reverse(diagonalize_rtm_right_gen_sym(ll; bring_right_gen=true, normalize_factor))

for jj in eachindex(eigs_l)
    eL = eigs_l[jj]
    eR = eigs_r[jj]
    @test eL[abs.(eL) .> 1e-13] ≈ eR[abs.(eR) .> 1e-13]
    #@info "$(jj)"
end

end

@testset "Test whether computed entropies match " begin

s = siteinds(4, 20)

ll = random_mps(ComplexF64, s, linkdims=40)
llc = deepcopy(ll)
normalize_factor = rand()

eigs_r = diagonalize_rtm_right_gen_sym(ll; bring_right_gen=true, normalize_factor)

allents = build_entropies(eigs_r)

r2_cut = rtm2_contracted(ll, ll, 10)

@test abs(inner(ll,llc) - 1) < 1e-12

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