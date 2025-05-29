"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED canonical form, by default bring the MPS to it
    By default, normalizes the eigenvalues of the symmetric RTM. 
"""
function generalized_vn_entropy_symmetric(psiL::MPS; bring_left_gen::Bool=true, normalize_eigs::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    gen_ents = ComplexF64[]

    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the operator/final state side (for me that's on the left)
    for ii = mpslen:-1:2
        Ai = psiL[ii]
        #Bi = prime(Ai, "v") # consistent with the label assigned by generalized canon form
        #Bi = prime(Ai, linkinds(psiL,ii), linkinds(psiL,ii-1)) 
        Bi = psiR[ii]
        #right_env = ( Ai * Bi * right_env ) 
        right_env = Ai * right_env 
        right_env = Bi * right_env

        @assert order(right_env) == 2 
        #println(left_env)
        eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2])
        #gen_ent_cut = sum(eigss.*log.(eigss))
        
        
        if normalize_eigs
            eigss = eigss/sum(eigss) 
        else # If we don't normalize, warn if normalization is off
            if abs(sum(eigss) - 1.) > 0.01
                @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
            end
        end

        gen_ent_cut = ComplexF64(0.)
        for n=1:dim(eigss, 1)
            p = eigss[n,n]        # I don't think we need the ^2 here 
            gen_ent_cut -= p * log(p)
            #println("[$ii]temp = $(gen_ent_cut) | sum = $(sum(eigss))")
        end

        push!(gen_ents, gen_ent_cut)
    
    end

    return gen_ents
    
end

"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED canonical form 
"""
function generalized_svd_vn_entropy_symmetric(psi::MPS)

    _, ents, _ = truncate_rsweep_sym(psi; cutoff=1e-12, chi_max=maxlinkdim(psi), method="SVD")
     
    return real(ents)

end



function generalized_entropies_symmetric(psiL::MPS; which_ents::Vector, bring_left_gen::Bool=false)
 
    eigenvalues_rtm = diagonalize_rtm_left_gen_sym(psiL; bring_left_gen)
    gen_ents = build_entropies(eigenvalues_rtm, which_ents)
    return gen_ents
    
end

""" We can compute the "SVD" VN entropy by just doing a right (generalized) sweep """
function generalized_svd_vn_entropy(psi::MPS, phi::MPS)
    truncp = TruncParams(1e-12, maxlinkdim(psi)+maxlinkdim(phi))
    _, _, ents = truncate_rsweep(psi, phi, truncp)
    return real(ents)
end


function diagonalize_rtm_symmetric(psiL::MPS; bring_left_gen::Bool=true, normalize_eigs::Bool=true, sort_by_largest::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    eigenvalues_rtm = []

    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the operator/final state side (for me that's on the left)
    for ii = mpslen:-1:2
        Ai = psiL[ii]
        #Bi = prime(Ai, "v") # consistent with the label assigned by generalized canon form
        #Bi = prime(Ai, linkinds(psiL,ii), linkinds(psiL,ii-1)) 
        Bi = psiR[ii]
        #right_env = ( Ai * Bi * right_env ) 
        right_env = Ai * right_env 
        right_env = Bi * right_env

        @assert order(right_env) == 2 
        #println(left_env)
        eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2], sortby=abs, rev=true)
        #gen_ent_cut = sum(eigss.*log.(eigss))
        
        
        if normalize_eigs
            eigss = eigss/sum(eigss) 
        else # If we don't normalize, warn if normalization is off
            if abs(sum(eigss) - 1.) > 0.01
                @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
            end
        end

        if sort_by_largest
            eigss = sort(eigss, by=abs, rev=true)
        end

        push!(eigenvalues_rtm, eigss)
    
    end

    return eigenvalues_rtm
    
end
