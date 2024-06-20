using ITensors
using ITensorMPS
using ITransverse
using Test

@testset "Testing right gauge symmetric RTM diagonalization" begin

psi = myrMPS(20, 30)

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