
function check_equivalence_svd(left_mps::MPS, right_mps::MPS)

    mpslen = length(left_mps)
    @info "Checking if SVs are the same "
    @show mpslen

    # bring to "standard" right canonical forms individually - ortho center on the 1st site 
    # making copies along the way 

    L_ortho = orthogonalize(left_mps,  1, normalize=false)
    R_ortho = orthogonalize(right_mps, 1, normalize=false)

    # # ! does this change anything ? doesn't seem like it 
    # L_ortho = orthogonalize(left_mps,  mpslen)
    # R_ortho = orthogonalize(right_mps, mpslen)
    # orthogonalize!(L_ortho,1)
    # orthogonalize!(R_ortho,1)
    # normalize!(L_ortho)
    # normalize!(R_ortho)
    

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = Vector{ComplexF64}()

    # Left sweep with truncation 

    Ai = XUinv * L_ortho[1]
    Bi = XVinv * R_ortho[1] 

    # Generalized canonical - no complex conjugation!
    left_env = deltaS * Ai 
    left_env *= Bi 

    u1,s1,v1 = svd(left_env, ind(left_env,1); cutoff=1e-14, maxdim=20)

    linds = []
    full_mps = left_env
    for jj = 2:mpslen
        full_mps *= L_ortho[jj] 
        full_mps *= prime(R_ortho[jj], "Site")
        push!(linds, siteind(L_ortho,jj))
    end

    @show linds
    @show inds(full_mps)


    u2,s2,v2 = svd(full_mps, linds; cutoff=nothing, maxdim=20)

    @show s1
    @show s2

end
