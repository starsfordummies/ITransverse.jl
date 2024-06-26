"""
Diagonalize RTM for a *symmetric* environment (psiL,psiL) at a given `cut`
    Assuming we're in LEFT GENERALIZED SYMMETRIC canonical form, we build *right* environments 
    for the RTM and diagonalize them:

    `   
    |  |  |  |  |   _____                ____
    D--D--D--D--D--|     |   (same λ) --|    |
                   | renv|      ≃       |    |
    D--D--D--D--D--|_____|            --|____|
    |  |  |  |  |

    `
"""
function diagonalize_rtm_left_gen_sym(psiL::MPS, cut::Int; bring_left_gen::Bool=false)

    @assert cut > 1 
    @assert cut < length(psiL)

    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4
        @warn" overlap not 1: $overlap"
    end
    
    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the right 
    for ii = mpslen:-1:cut
        Ai = psiL[ii]
        Bi = psiR[ii]
        right_env = Ai * right_env 
        right_env = Bi * right_env

    end

    @assert order(right_env) == 2 
    #println(left_env)
    eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2])
    
    if abs(sum(eigss) - 1.) > 0.01
        @warn "RTM not well normalized? Σeigs-1 = $(abs(sum(eigss) - 1.)) "
    end

    return diag(matrix(eigss))
    
end



"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED canonical form 
    Returns a list of vectors of eigenvalues, one for each cut 
"""
function diagonalize_rtm_left_gen_sym(psiL::MPS; bring_left_gen::Bool=false, normalize_factor::Number=1.0)

    psi_gauged = psiL/normalize_factor
    
    # we can enforce to bring it in left symmetric gen. canonical form 
    if bring_left_gen
        psi_gauged = gen_canonical_left(psi_gauged)
    end

    mpslen = length(psi_gauged)

    overlap = overlap_noconj(psi_gauged,psi_gauged)
    if abs(1-overlap) > 1e-4
        @warn "overlap not 1: $(overlap)"
    end
    
    eigs_rtm_t = []

    right_env = ITensor(1.)

    psiR = prime(linkinds, psi_gauged)

    # Start from the right 
    for ii = mpslen:-1:2
        Ai = psi_gauged[ii]
        Bi = psiR[ii]

        right_env = Ai * right_env 
        right_env = Bi * right_env

        @assert order(right_env) == 2 
        eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2])
        
        if abs(sum(eigss) - 1.) > 0.01
            @warn "RTM not well normalized? Σeigs-1=$(abs(sum(eigss) - 1.)) "
        end

        push!(eigs_rtm_t, diag(matrix(eigss)))
    
    end

    return eigs_rtm_t
    
end








""" Given `psi` MPS, build the symmetric RTM <psibar|psi>:
Bring psi in generalized RIGHT symmetric canonical form,
then contract LEFT enviroments and diagonalize them.
Returns an array of arrays containing the eigenvalues of the RTM for each cut """
function diagonalize_rtm_right_gen_sym(psi::MPS; bring_right_gen::Bool=false, normalize_factor::Number=1.0)

    mpslen = length(psi)

    psi_gauged = psi/normalize_factor

    if bring_right_gen
        psi_gauged = gen_canonical_right(psi)
    end

    lenv= ITensor(1.)
    s = siteinds(psi_gauged)

    eigs_rtm_t = []


    # if abs(sum(eigss) - 1.) > 0.01
    #     @warn "RTM not well normalized? Σeigs-1=$(abs(sum(eigss) - 1.)) "
    # end

    for jj in 1:mpslen-1

        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )

        @assert ndims(lenv) == 2 

        eigss, _ = eigen(lenv, inds(lenv)[1],inds(lenv)[2])
        
        push!(eigs_rtm_t, diag(matrix(eigss)))

    end
    
    return eigs_rtm_t
end
