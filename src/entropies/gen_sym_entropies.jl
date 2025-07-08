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

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    gen_ents = Vector{ComplexF64}(undef, mpslen-1) 

    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the *right* (operator side)
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

        gen_ents[ii-1] = gen_ent_cut
    
    end

    return gen_ents
    
end

function generalized_vn_entropy_symmetric(psiL::MPS, cut::Int; bring_left_gen::Bool=true, normalize_eigs::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    psiR = prime(linkinds, psiL)


    # Build right_env up to 'cut'
    right_env = ITensor(1.)

    # Start from the *right* (operator side)
    for ii = mpslen:-1:cut
        right_env = psiL[ii] * right_env 
        right_env = psiR[ii] * right_env
    end

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
        p = eigss[n,n]  
        gen_ent_cut -= p * log(p)
    end

    return gen_ent_cut
    
end


function generalized_r2_entropy_symmetric(psiL::MPS; bring_left_gen::Bool=true, normalize_eigs::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    gen_ents = Vector{ComplexF64}(undef, mpslen-1) 

    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the *right* (operator side)
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
            gen_ent_cut += p^2
            #println("[$ii]temp = $(gen_ent_cut) | sum = $(sum(eigss))")
        end

        gen_ents[ii-1] = gen_ent_cut
    
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



""" We can compute the "SVD" VN entropy by just doing a right (generalized) sweep """
function generalized_svd_vn_entropy(psi::MPS, phi::MPS)
    truncp = TruncParams(1e-12, maxlinkdim(psi)+maxlinkdim(phi))
    _, _, ents = truncate_rsweep(psi, phi, truncp)
    return real(ents)
end

