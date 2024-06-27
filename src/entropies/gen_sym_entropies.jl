


"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    at a given cut, assuming we're in *LEFT* mixed/generalized canonical form 
"""
function generalized_entropy_symmetric(psiL::MPS, cut::Int)

    mpslen = length(psiL)
    #links = linkinds(psiL)

    right_env = 1. # ITensor(1.)

    for ii = mpslen:-1:cut 
        Ai = psiL[ii]
        Bi = prime(Ai, "Link,v") # This is consistent with the label assigned by my algo for generalized canon form
        right_env = Ai * right_env 
        right_env = Bi * right_env 
    
    end
    eigss, _ = eigen(right_env)

    if abs(sum(eigss) - 1.) > 0.01
        @warn "RTM not well normalized? Σeigs=1-$(abs(sum(eigss) - 1.)) "
    end

    gen_ent_cut = log(sum(eigss))

    return gen_ent_cut
    
end

"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED canonical form 
"""
function generalized_entropy_symmetric(psiL::MPS; bring_left_gen::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4
        @warn" overlap not 1: $overlap"
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
        
        if abs(sum(eigss) - 1.) > 0.01
            @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
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


""" TODO these two are the same! """

function generalized_entropy_symmetric_new(psiL::MPS; bring_left_gen::Bool=false)
 
    eigenvalues = diagonalize_rtm_left_gen_sym(psiL; bring_left_gen)
    gen_ents = build_entropies(eigenvalues)
    return gen_ents
    
end



function build_entropies(psiL::MPS; bring_left_gen::Bool=false, sweep_direction::String="R")
    if sweep_direction == "R"
    spectra = diagonalize_rtm_left_gen_sym(psiL, bring_left_gen=bring_left_gen)
    end
    return build_entropies(spectra)
end
