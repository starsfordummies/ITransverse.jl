""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation.
Returns 
1,2) copies of the two input MPS
3) an effective entropy computed from the SV of the environments
4) the overlap between the two 
"""

function truncate_lsweep(psi::MPS, phi::MPS; cutoff::Real, chi_max::Int)

    elt = eltype(psi[1])
    mpslen = length(phi)

    psi_ortho = orthogonalize(psi, 1)
    phi_ortho = orthogonalize(phi, 1)

    XUinv, XVinv, left_env = (togpu(ITensor(elt,1.)), togpu(ITensor(elt,1.)), togpu(ITensor(elt,1.)))

    ents_sites = ComplexF64[]

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        left_env *= Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)

        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag

        left_env *= XU
        left_env *= XV

        psi_ortho[ii] = Ai * XU  
        phi_ortho[ii] = Bi * XV

        push!(ents_sites, scalar(tocpu((-S*log.(S)))))
      
    end

    # the last two 
    psi_ortho[end] = XUinv * psi_ortho[end]
    phi_ortho[end] = XVinv * phi_ortho[end]

    gen_overlap = scalar(tocpu((left_env * ( psi_ortho[end] *  phi_ortho[end] ) )))

    return psi_ortho, phi_ortho, ents_sites, gen_overlap

end





"""
Truncates optimizing the overlap (L|R), returns new L and R in generalized orthogonal form 
and (an estimate for) the gen. entropy at each site.
Cutoff on SV of the transition matrix, given by `cutoff` param

For this, we first bring to the usual *right* canonical form both MPS (ortho center on 1st site), \\
then we build environments L|R from the *left* and truncate on their SVDs (or EIGs depending on `method`)

So this can be seen as a "RL: Right(can)Left(gen)" sweep 
"""



""" Brings to right generalized canonical form two MPS `psi` and `phi`, truncating along the way if necessary.
Returns

1) updated `psi`
2) updated `phi` 
3) effective entropies calculated form the SVD of the environments
4) overlap (psi|phi) [without conjugating psi] """
function truncate_rsweep(psi::MPS, phi::MPS; cutoff::Real, chi_max::Int)

    elt = eltype(psi[1])
    mpslen = length(psi)

    # first bring to left canonical form 
    psi_ortho = orthogonalize(psi, mpslen, normalize=false)
    phi_ortho = orthogonalize(phi, mpslen, normalize=false)

    XUinv, XVinv, right_env = (togpu(ITensor(elt, 1.)), togpu(ITensor(elt, 1.)), togpu(ITensor(elt, 1.))) 
    
    ents_sites = ComplexF64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        right_env *= Ai 
        right_env *= Bi 

        rnorm = norm(right_env)
        #@show ii, rnorm

        if rnorm > 1e6 || rnorm < 1e-6
            @warn "Norm of environment is $(rnorm), watch for roundoff errs"
        end

        #right_env /= rnorm
        
        @assert order(right_env) == 2

        U,S,Vdag = svd(right_env, ind(right_env,1); cutoff=cutoff, maxdim=chi_max)

        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag

        right_env *= XU
        right_env *= XV

        # Set updated matrices
        psi_ortho[ii] = Ai * XU  
        phi_ortho[ii] = Bi * XV

        push!(ents_sites, scalar(tocpu((-S*log.(S)))))

    end

    # the final two
    psi_ortho[1] = XUinv * psi_ortho[1]
    phi_ortho[1] = XVinv * phi_ortho[1]

    gen_overlap = scalar(tocpu((right_env * ( phi_ortho[1] *  psi_ortho[1] ) )))


    return psi_ortho, phi_ortho, ents_sites, gen_overlap

end



function truncate_normalize_lsweep(psi::MPS, phi::MPS, truncp::trunc_params)
    psi_n, phi_n, ee, ov = truncate_lsweep(psi, phi, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    ov_alt =  overlap_noconj(psi_n,phi_n)
    if abs(ov - ov_alt) > 0.1
        @warn "Check canonical form!? $(ov) vs $(ov_alt)"
    end
    psi_n[end] /= sqrt(ov_alt)
    phi_n[end] /= sqrt(ov_alt)
    
    return psi_n, phi_n, [e./sum(e) for e in ee]
end

function truncate_normalize_rsweep(psi::MPS, phi::MPS, truncp::trunc_params)
    psi_n, phi_n, ee, ov = truncate_rsweep(psi, phi, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    ov_alt =  overlap_noconj(psi_n,phi_n)
    if abs(ov - ov_alt) > 0.1
        @warn "Check canonical form!? $(ov) vs $(ov_alt)"
    end
    psi_n[1] /= sqrt(ov_alt)
    phi_n[1] /= sqrt(ov_alt)

    return psi_n, phi_n, [e./sum(e) for e in ee]
end