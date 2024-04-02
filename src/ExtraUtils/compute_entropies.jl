#using ITensors
#include("./utils.jl")

""" Computes the Von Neumann entanglement entropy of an MPS psi at a given cut
"""
function vn_entanglement_entropy_cut(psi::MPS, cut::Int)

    orthogonalize!(psi, cut)
    #println(norm(psi))

    if cut == 1
        _,S,_ = svd(psi[cut], (siteind(psi,cut)))
    else
        _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
    end

    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end

    return SvN
end



""" Computes the Von Neumann entanglement entropy of an MPS psi at all links, 
returns a vector of floats containing the VN entropies 
"""
function vn_entanglement_entropy(psi::MPS)

    workpsi = normalize(psi)

    SvNs = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = vn_entanglement_entropy_cut(workpsi, icut)
        push!(SvNs, Si)
    end

    return SvNs
end



function renyi_entanglement_entropy_cut(psi::MPS, cut::Int, nren::Int)

    S_ren = 0.0

    if nren == 1  # VN entropy
        S_ren = vn_entanglement_entropy_cut(psi, cut)

    elseif nren == 2 # Renyi 2 
            
        orthogonalize!(psi, cut)
        #println(norm(psi))

        if cut == 1
            _,S,_ = svd(psi[cut], (siteind(psi,cut)))
        else
            _,S,_ = svd(psi[cut], (linkind(psi, cut-1), siteind(psi,cut)))
        end

        sum_s2 = 0.0
        for n=1:dim(S, 1)
            p = S[n,n]^2
            sum_s2 += p^2
        end
        S_ren = -log(sum_s2)

    else # Not implemented yet
        S_ren = 0.
    end

    return S_ren
end



""" Computes the nth Renyi entanglement entropy of an MPS psi at all links, 
returns a vector of floats containing the VN entropies 
"""
function renyi_entanglement_entropy(psi::MPS, nren::Int=2)

    workpsi = normalize(psi)

    SvNs = Vector{Float64}()

    for icut=1:length(workpsi)-1
        Si = renyi_entanglement_entropy_cut(workpsi, icut, nren)
        push!(SvNs, Si)
    end

    return SvNs
end



"""
Generalized entropy for a symmetric environment (psiL,psiL)
    at a given cut 
"""
function generalized_entropy_symmetric_cut(psiL::MPS, cut::Int)
    # Assuming we're dealing with a SYMMETRIC transition matrix (ie. L=R)
    # (TODO: up to transposes?))

    # Assuming we're in LEFT mixed/generalized canonical form 

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
        print("warning, RTM not well normalized? $(abs(sum(eigss) - 1.)) ")
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
        println("Warning: overlap not 1: $overlap")
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
            print("warning, RTM not well normalized? $(abs(sum(eigss) - 1.)) ")
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
Generalized entropy for two vectors (psiL, psiR) \\
Assuming we're in LEFT GENERALIZED canonical form 
"""
function generalized_entropy(psiL::MPS,psiR::MPS)

    # Assuming we're in LEFT mixed/generalized canonical form 

    mpslen = length(psiL)
    #links = linkinds(psiL)

    println("overlap = $(overlap_noconj(psiL,psiR))")
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
            print("warning, RTM not well normalized? $(abs(sum(eigss) - 1.)) ")
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


