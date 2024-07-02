""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation """

function truncate_lsweep(right_mps::MPS, left_mps::MPS; cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    R_ortho = orthogonalize(right_mps, 1)
    L_ortho = orthogonalize(left_mps,  1)

    XUinv, XVinv, left_env = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = ComplexF64[]

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

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

        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        push!(ents_sites, log(sum(S)))
      
    end

    # the last two 
    L_ortho[end] = XUinv * L_ortho[end]
    R_ortho[end] = XVinv * R_ortho[end]

    gen_overlap = scalar(left_env * ( L_ortho[end] *  R_ortho[end] ) )

    return L_ortho, R_ortho, ents_sites, gen_overlap

end





"""
Truncates optimizing the overlap (L|R), returns new L and R in generalized orthogonal form 
and (an estimate for) the gen. entropy at each site.
Cutoff on SV of the transition matrix, given by `cutoff` param

For this, we first bring to the usual *right* canonical form both MPS (ortho center on 1st site), \\
then we build environments L|R from the *left* and truncate on their SVDs (or EIGs depending on `method`)

So this can be seen as a "RL: Right(can)Left(gen)" sweep 
"""



""" Brings to right generalized canonical form two MPS, truncating along the way if necessary.
Returns updated R, L and effective entropies calculated form the SVD of the environments """
function truncate_rsweep(right_mps::MPS, left_mps::MPS; cutoff::Real, chi_max::Int)

    mpslen = length(right_mps)

    # first bring to left canonical form 
    R_ortho = orthogonalize(right_mps, mpslen, normalize=false)
    L_ortho = orthogonalize(left_mps,  mpslen, normalize=false)

    XUinv, XVinv, right_env = (ITensor(1.), ITensor(1.), ITensor(1.)) 
    
    ents_sites = ComplexF64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * R_ortho[ii]
        Bi = XVinv * L_ortho[ii] 

        right_env *= Ai 
        right_env *= Bi 

        rnorm = norm(right_env)

        if rnorm > 1e6 || rnorm < 1e-6
            @warn "Norm of environment is $(rnorm), watch for roundoff errs"
        end

        right_env /= rnorm
        
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
        R_ortho[ii] = Ai * XU  
        L_ortho[ii] = Bi * XV

        push!(ents_sites, log(sum(S)))

    end

    # the final two
    R_ortho[1] = XUinv * R_ortho[1]
    L_ortho[1] = XVinv * L_ortho[1]

    return R_ortho, L_ortho, ents_sites

end



function truncate_normalize_lsweep(right_mps::MPS, left_mps::MPS, truncp::trunc_params)
    ll, rr, ee = truncate_lsweep(right_mps, left_mps, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    LR =  overlap_noconj(ll,rr)
    rr[end] /= sqrt(LR)
    ll[end] /= sqrt(LR)
    
    return rr,ll, [e./sum(e) for e in ee]
end

function truncate_normalize_rsweep(right_mps::MPS, left_mps::MPS, truncp::trunc_params)
    rr, ll, ee = truncate_rsweep(right_mps, left_mps, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    LR =  overlap_noconj(rr,ll)
    rr[1] /= sqrt(LR)
    ll[1] /= sqrt(LR)

    return rr, ll, [e./sum(e) for e in ee]
end