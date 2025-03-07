""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation.
Returns 
1,2) copies of the two input MPS
3) an effective entropy computed from the SV of the environments
4) the overlap between the two 
"""
function truncate_lsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    truncate_lsweep(psi, phi; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
end

function truncate_lsweep(psi::MPS, phi::MPS; cutoff::Real, chi_max::Int)

    #elt = eltype(psi[1])
    mpslen = length(phi)

    psi_ortho = orthogonalize(psi, 1)
    phi_ortho = orthogonalize(phi, 1)

    XUinv, XVinv, left_env = (ITensor(1),ITensor(1),ITensor(1))

    #ents_sites = ComplexF64[]
    ents_sites = Vector{Float64}(undef, mpslen - 1) 

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


        Snorm = tocpu(S./sum(S))
        ents_sites[ii] = scalar((-Snorm*log.(Snorm)))

        #push!(ents_sites, scalar(tocpu((-S*log.(S)))))
      
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
function truncate_rsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    truncate_rsweep(psi, phi; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
end
function truncate_rsweep(psi::MPS, phi::MPS; cutoff::Real=1e-12, chi_max::Int=max(maxlinkdim(psi),maxlinkdim(phi)))

    #elt = eltype(psi[1])
    mpslen = length(psi)

    # first bring to left canonical form  
    psi_ortho = orthogonalize(psi, mpslen)
    phi_ortho = orthogonalize(phi, mpslen)

    XUinv, XVinv, right_env = (ITensor(1), ITensor(1), ITensor(1))
    
    # For the non-symmetric case we can only truncate with SVD, so ents will be real 
    ents_sites = fill(0., mpslen-1)  # Float64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        right_env *= Ai 
        right_env *= Bi 

        #rnorm = norm(right_env)
        #@show ii, rnorm

        # TODO maybe we should correct along the way if envs become too large
        # if rnorm > 1e14 || rnorm < 1e-14
        #     @warn "Norm of environment $(ii) is $(rnorm), watch for roundoff errs"
        #     @warn overlap_noconj(psi,phi)
        #     @warn inner(psi,phi)
        #     @warn norm(psi)
        #     @warn norm(phi)
        # end

        #right_env /= rnorm
        
        @assert order(right_env) == 2

        #U,S,Vdag = svd(right_env, ind(right_env,1); cutoff=cutoff, maxdim=chi_max)
        U,S,Vdag = matrix_svd(right_env; cutoff=cutoff, maxdim=chi_max)

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

        Snorm = tocpu(S./sum(S))
        ents_sites[ii-1] = scalar((-Snorm*log.(Snorm)))

        #@info "setting psi[$(ii)]"

    end

    # the final two
    psi_ortho[1] = XUinv * psi_ortho[1]
    phi_ortho[1] = XVinv * phi_ortho[1]

    gen_overlap = scalar(tocpu((right_env * ( phi_ortho[1] *  psi_ortho[1] ) )))


    return psi_ortho, phi_ortho, ents_sites, gen_overlap

end



function truncate_normalize_lsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    psi_n, phi_n, ee, ov = truncate_lsweep(psi, phi, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    ov_alt =  overlap_noconj(psi_n,phi_n)
    if abs(ov - ov_alt) > 0.1
        @warn "Check canonical form!? $(ov) vs $(ov_alt)"
    end
    psi_n[end] /= sqrt(ov_alt)
    phi_n[end] /= sqrt(ov_alt)
    
    return psi_n, phi_n, [e./sum(e) for e in ee]
end

function truncate_normalize_rsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    psi_n, phi_n, ee, ov = truncate_rsweep(psi, phi, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    ov_alt =  overlap_noconj(psi_n,phi_n)
    if abs(ov - ov_alt) > 0.1
        @warn "Check canonical form!? $(ov) vs $(ov_alt)"
    end
    psi_n[1] /= sqrt(ov_alt)
    phi_n[1] /= sqrt(ov_alt)

    return psi_n, phi_n, [e./sum(e) for e in ee]
end


""" Generic sweep, calls left or right according to `truncp.direction` """
function truncate_sweep(psi::MPS, phi::MPS, truncp::TruncParams)
    if truncp.direction == "left"
        #@info "initial state side sweep"
        truncate_lsweep(psi, phi, truncp)
    elseif truncp.direction == "right"
        #@info "operator state side sweep"
        truncate_rsweep(psi, phi, truncp)
    else
        @error "Sweep direction should be left|right"
    end
end

""" Generic sweep, calls left or right according to `truncp.direction` """
function truncate_normalize_sweep(psi::MPS, phi::MPS, truncp::TruncParams)
    if truncp.direction == "left"
        #@info "initial state side sweep"
        truncate_normalize_lsweep(psi, phi, truncp)
    elseif truncp.direction == "right"
        #@info "operator state side sweep"
        truncate_normalize_rsweep(psi, phi, truncp)
    else
        @error "Sweep direction should be left|right"
    end
end

""" We can compute the "SVD" VN entropy by just doing a right (generalized) sweep """
function generalized_svd_vn_entropy(psi, phi)
    truncp = TruncParams(1e-14, maxlinkdim(psi))
    _, _, ents, _ = truncate_rsweep(psi, phi, truncp)
    return ents
end


""" Attempt at making a faster sweep with little success"""
function truncate_rsweep!(psi::MPS, phi::MPS; cutoff::Real=1e-12, chi_max::Int=max(maxlinkdim(psi),maxlinkdim(phi)))

    #elt = eltype(psi[1])
    mpslen = length(psi)

    # first bring to left canonical form  
    orthogonalize!(psi, mpslen)
    orthogonalize!(phi, mpslen)

    XUinv, XVinv, right_env = (ITensor(1), ITensor(1), ITensor(1))
    
    # For the non-symmetric case we can only truncate with SVD, so ents will be real 
    ents_sites = fill(0., mpslen-1)  # Float64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * psi[ii]
        Bi = XVinv * phi[ii] 

        right_env *= Ai 
        right_env *= Bi 

        @assert order(right_env) == 2

        #U,S,Vdag = svd(right_env, ind(right_env,1); cutoff=cutoff, maxdim=chi_max)
        U,S,Vdag = matrix_svd(right_env; cutoff=cutoff, maxdim=chi_max)

        
        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag

        # right_env *= XU
        # right_env *= XV
        right_env = delta(only(uniqueinds(XU, right_env)), only(uniqueinds(XV, right_env)))

        # Set updated matrices
        psi[ii] = Ai * XU  
        phi[ii] = Bi * XV

        Snorm = tocpu(normalize(S))
        ents_sites[ii-1] = scalar((-Snorm*log.(Snorm)))

        #@info "setting psi[$(ii)]"

    end

    # the final two
    psi[1] = XUinv * psi[1]
    phi[1] = XVinv * phi[1]

    gen_overlap = scalar(tocpu((right_env * ( phi[1] *  psi[1] ) )))


    return ents_sites, gen_overlap

end


