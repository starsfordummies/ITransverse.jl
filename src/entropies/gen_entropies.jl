#############################################
####### VN ENTROPIES  ########################
#############################################


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





""" For each site build a series of entropies given the eigenvalues"""
function build_entropies(spectra::Vector)
    vns = []
    tsallis2 = []
    renyi2s = [] 
    traces_Tt2 = []
    for eigss in spectra
        vn = 0.
        r2 = 0.
        for n in eachindex(eigss)
            p = eigss[n]       # I don't think we need the ^2 here 
            if abs(p) > 1e-15
                vn -= p * log(p)
                r2 += p^2
            end
        end
        push!(vns, vn)
        push!(tsallis2, - (r2 - 1.))
        push!(renyi2s, - log(r2))
        push!(traces_Tt2, r2)

    end

    entropies = Dict(:vn=>vns, :tsallis2=>tsallis2, :renyi2=>renyi2s, :traces_Tt2=>traces_Tt2)
    return entropies
end

