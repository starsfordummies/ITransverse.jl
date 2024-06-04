#############################################
#######! VN ENTROPY  ########################
#############################################

"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    at a given cut, assuming we're in *LEFT* mixed/generalized canonical form 
"""
function generalized_entropy_symmetric_cut(psiL::MPS, cut::Int)

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
function generalized_entropy_symmetric(psiL::MPS, bring_left_gen::Bool=false)
    # Assuming we're dealing with a SYMMETRIC transition matrix (ie. L=R)
    # (TODO: up to transposes?))

    # Assuming we're in LEFT mixed/generalized canonical form ?
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


""" Backwards compatibility """
function generalized_entropy(psiL::MPS,psiR::MPS)
    generalized_vn_entropy(psiL,psiR)
end


""" 
Generalized entropy for two vectors (psiL, psiR) \\
Assuming we're in LEFT GENERALIZED canonical form 
"""
function generalized_vn_entropy(psiL::MPS,psiR::MPS)

    # check gen form 
    check_gencan_left_phipsi(psiL,psiR)

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiR)
    if abs(1-overlap) > 1e-4
        @warn" overlap not 1: $overlap"
    end

    gen_ents = Vector{ComplexF64}()

    right_env = ITensor(1.)

    for ii = mpslen:-1:2
        Ai = psiL[ii]
        Bi = psiR[ii]
        right_env = Ai * right_env 
        right_env = Bi * right_env
        @assert order(right_env) == 2 
        #println(left_env)
        eigss, _ = eigen(right_env, inds(right_env)[1], inds(right_env)[2])
        
        gen_ent_cut = 0.
        if abs(sum(eigss) - 1.) > 0.01
            @warn "RTM not well normalized? 1-Σeigs = $(abs(sum(eigss) - 1.)) "
        end

        for n=1:dim(eigss, 1)
            p = eigss[n,n]  # I don't think we need the ^2 here 
            gen_ent_cut -= p * log(p)
            #println("[$ii]temp = $(gen_ent_cut) | sum = $(sum(eigss))")
        end
        push!(gen_ents, gen_ent_cut)
    
    end

    return gen_ents
    
end

function generalized_renyi_entropy(psiL::MPS,psiR::MPS,n::Int; normalize::Bool=false)

    if normalize
        psiL = deepcopy(psiL)
        psiR = deepcopy(psiR)
        overlap = overlap_noconj(psiL,psiR)
        psiL[end] /= sqrt(overlap)
        psiR[end] /= sqrt(overlap)
    end

    if n == 1
        return generalized_entropy(psiL,psiR)
    end

    # check gen form 
    @info "checking form" 
    sleep(2)
    check_gencan_left_phipsi(psiL,psiR)

    
    # Assuming we're in LEFT mixed/generalized canonical form 

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiR)
    if abs(1-overlap) > 1e-4
        @warn" overlap not 1: $overlap"
    end

    #println("overlap = $(overlap_noconj(psiL,psiR))")

    renyi_gen_ents = Vector{ComplexF64}()

    right_env = ITensor(1.)

    for ii = mpslen:-1:2
        Ai = psiL[ii]
        Bi = psiR[ii]
        right_env = Ai * right_env 
        right_env = Bi * right_env
        @assert order(right_env) == 2 
        #println(left_env)
        eigss, _ = eigen(right_env, inds(right_env)[1], inds(right_env)[2])
        
        renyi_gen_ent_cut = 0.

        if abs(sum(eigss) - 1.) > 0.01
            @warn "[$(ii)/$(mpslen)] RTM not well normalized? $(abs(sum(eigss) - 1.)) "
        # else
        #     @info "RTM well normalized"
        end

        for jj=1:dim(eigss, 1)
            p = eigss[jj,jj]  
            renyi_gen_ent_cut += p^n
            #println("[$ii]temp = $(gen_ent_cut) | sum = $(sum(eigss))")
        end
        renyi_gen_ent_cut /= 1-n

        push!(renyi_gen_ents, renyi_gen_ent_cut)
    
    end

    return renyi_gen_ents
    
end







"""
Diagonalize RTM for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED SYMMETRIC canonical form 
"""
function gen_symm_diagonalize_rtm(psiL::MPS, cut::Int; bring_left_gen::Bool=false)

    @assert cut > 1 
    @assert cut < length(psiL)
    # Assuming we're in LEFT mixed/generalized canonical form ?
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
    for ii = mpslen:-1:cut
        Ai = psiL[ii]
        Bi = psiR[ii]
        #right_env = ( Ai * Bi * right_env ) 
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
function gen_symm_diagonalize_rtm(psiL::MPS; bring_left_gen::Bool=false)

    # Assuming we're in LEFT mixed/generalized canonical form ?
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4
        @warn "overlap not 1: $(overlap)"
    end
    
    eigs_rtm_t = []

    right_env = ITensor(1.)

    psiR = prime(linkinds, psiL)

    # Start from the operator/final state side (for me that's on the left)
    for ii = mpslen:-1:2
        Ai = psiL[ii]
        Bi = psiR[ii]

        right_env = Ai * right_env 
        right_env = Bi * right_env

        @assert order(right_env) == 2 
        #println(left_env)
        eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2])
        #gen_ent_cut = sum(eigss.*log.(eigss))
        
        if abs(sum(eigss) - 1.) > 0.01
            @warn "RTM not well normalized? Σeigs-1=$(abs(sum(eigss) - 1.)) "
        end


        push!(eigs_rtm_t, diag(matrix(eigss)))
    
    end

    return eigs_rtm_t
    
end

""" For each site build a series of entropies given the eigenvalues"""
function build_entropies(spectra::Vector)
    vns = []
    tsallis2 = []
    renyi2s = [] 
    for eigss in spectra
        vn = 0.
        r2 = 0.
        for n in eachindex(eigss)
            p = eigss[n]       # I don't think we need the ^2 here 
            vn -= p * log(p)
            r2 = p^2
        end
        push!(vns, vn)
        push!(tsallis2, - (r2 - 1.))
        push!(renyi2s, - log(r2))
    end

    entropies = Dict(:vn=>vns, :tsallis2=>tsallis2, :renyi2=>renyi2s)
    return entropies
end


function build_entropies(psiL::MPS; bring_left_gen::Bool=false)
    spectra = gen_symm_diagonalize_rtm(psiL, bring_left_gen=bring_left_gen)
    return build_entropies(spectra)
end
