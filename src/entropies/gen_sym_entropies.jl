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

    right_env = ITensors.OneITensor() 

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

        #@assert order(right_env) == 2 
        #println(left_env)
        eigss = eigvals(right_env)
        #gen_ent_cut = sum(eigss.*log.(eigss))
        
        sumeigss = sum(eigss)
        
        if normalize_eigs
            eigss = eigss/sumeigss
        else # If we don't normalize, warn if normalization is off
            if abs(sumeigss- 1.) > 0.01
                @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
            end
        end

        # gen_ent_cut = ComplexF64(0.)
        # for n=1:dim(eigss, 1)
        #     p = eigss[n,n]        # I don't think we need the ^2 here 
        #     gen_ent_cut -= p * log(p)
        #     #println("[$ii]temp = $(gen_ent_cut) | sum = $(sum(eigss))")
        # end
        gen_ent_cut = -sum(p -> p * log(p), eigss)


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
    right_env = ITensors.OneITensor()

    # Start from the *right* (operator side)
    for ii = mpslen:-1:cut
        right_env = psiL[ii] * right_env 
        right_env = psiR[ii] * right_env
    end

    @assert order(right_env) == 2 
    #println(left_env)
    eigss = eigvals(right_env)
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

    right_env = ITensors.OneITensor()

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
        eigss = eigvals(right_env)
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

function vn_from_matrix(λ_matrix::Matrix{T}) where {T<:Real}
    λ_safe = max.(λ_matrix, eps(T))
    
    # Compute -λ*log(λ) element-wise, zeroing out tiny eigenvalues
    contrib = @. ifelse(λ_matrix > eps(T), -λ_matrix * log(λ_safe), 0.0)
    
    # Sum along columns (eigenvalues) for each row (bipartition)
    return vec(sum(contrib, dims=2))
end

function generalized_svd_vn_entropy_symmetric(psi::MPS)
    _, svs = truncate_rsweep_sym(psi; cutoff=1e-12, maxdim=maxlinkdim(psi), method="SVD")
    return vn_from_matrix(svs)
end



""" We can compute the "SVD" VN entropy by just doing a right (generalized) sweep """
function generalized_svd_vn_entropy(psi::MPS, phi::MPS)
    truncp = TruncParams(1e-12, maxlinkdim(psi)+maxlinkdim(phi))
    _, _, svs = truncate_rsweep(psi, phi, truncp)
    return vn_from_matrix(svs)
end


""" Computes the symmetric generalized entropies: Given an input MPS |psi>, diagonalizes the RTMs |psi><psi*| 
and builds alpha-order Renyi entropies as specified by `which_ents` (alpha=1 ie. VN is always computed). Returns a dict """
function gensym_renyi_entropies(psiL::MPS; which_ents=[0.5,1,2], bring_left_gen::Bool=true, normalize_eigs::Bool=true)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    eigs_rtm = Vector{Vector{ComplexF64}}(undef, mpslen-1)

    right_env = ITensors.OneITensor()

    psiR = prime(linkinds, psiL)

    # Start from the *right* (operator side) and vbuild spectra 
    for ii = mpslen:-1:2
        Ai = psiL[ii]

        Bi = psiR[ii]

        right_env = Ai * right_env 
        right_env = Bi * right_env

        @assert order(right_env) == 2 
        eigss = eigvals(right_env)
 
        if normalize_eigs
            eigss = eigss/sum(eigss) 
        end


        eigs_rtm[ii-1] = eigss
    
    end

    gen_ents = renyi_entropies(eigs_rtm; which_ents, normalize_eigs=false)

    return gen_ents
    
end


""" Computes the generalized SVD entropies: Given input MPS |phi> and <psi|, < diagonalizes the RTM |phi><psi| 
and builds alpha-order Renyi entropies as specified by `which_ents` (alpha=1 ie. VN is always computed). Returns a dict """
function gensvd_renyi_entropies(psi::MPS, phi::MPS; which_ents=[0.5,1,2], normalize_eigs::Bool=true)
 
    mpslen = length(psi)

    # first bring to left canonical form  
    psi_ortho = orthogonalize(psi, mpslen)
    phi_ortho = orthogonalize(phi, mpslen)

    XUinv, XVinv, right_env = (ITensors.OneITensor(), ITensors.OneITensor(), ITensors.OneITensor())
    
    # For the non-symmetric case we can only truncate with SVD, so ents will be real 
    svds_rdm = Vector{Vector{Float64}}(undef, mpslen-1)


    # Start from the *right* side and sweep towards the left 
    for ii in mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        right_env *= Ai 
        right_env *= Bi 

        @assert order(right_env) == 2

        U,S,Vdag = svd(right_env, ind(right_env,1); cutoff=1e-14)

        norm_factor = sum(S)
        Snorm = normalize_eigs ? S/norm_factor : S

        XU = dag(U)
        XUinv = U

        XV = dag(Vdag) 
        XVinv = Vdag

        right_env /= norm_factor
   
        right_env *= XU
        right_env *= XV

        svds_rdm[ii-1] = array(diag(Snorm))


    end

    gen_svd_ents = renyi_entropies(svds_rdm; which_ents, normalize_eigs=false)

    return gen_svd_ents
    
end


""" Computes the generalized SVD entropies: Given input MPS |phi> and <psi|, < diagonalizes the RTM |phi><psi| 
and builds alpha-order Renyi entropies as specified by `which_ents` (alpha=1 ie. VN is always computed). Returns a dict """
function gensvd_renyi_entropies(psi::MPS; which_ents=[0.5,1,2], normalize_eigs::Bool=true)
 
    mpslen = length(psi)
    elt = eltype(psi[1])
    sits = siteinds(psi)

    # first bring to left canonical form  
    psi_ortho = orthogonalize(psi, mpslen)

    XUinv, right_env = (ITensors.OneITensor(), ITensors.OneITensor(), ITensors.OneITensor())
    
    # For the non-symmetric case we can only truncate with SVD, so ents will be real 
    svds_rdm = Vector{Vector{Float64}}(undef, mpslen-1)


    # Start from the *right* side and sweep towards the left 
    for ii in mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]


        right_env *= Ai 
        right_env *= Ai'
        right_env *= delta(elt, sits[ii], sits[ii]')
        
        F = symm_svd(right_env, ind(right_env,1), cutoff=1e-14)
        U = F.U
        S = F.S

        @assert order(right_env) == 2

        norm_factor = sum(S)
        Snorm = normalize_eigs ? S/norm_factor : S

        XU = dag(U)
        XUinv = U

        right_env /= norm_factor
   
        right_env *= XU
        right_env *= XU' 

        svds_rdm[ii-1] = array(diag(Snorm))

    end

    gen_svd_ents = renyi_entropies(svds_rdm; which_ents, normalize_eigs=false)

    return gen_svd_ents
    
end

""" Slow compute generalized Renyi2 for symmetric case RTM (psi,psi) for an interval [iA-fA] """
function gen_renyi2_sym_interval_manual(psi::MPS, iA::Int, fA::Int)

    LL = length(psi)

    psip = prime(linkinds, psi)
    lenv = ITensor(1)
    for kk = 1:iA-1
        lenv *= psi[kk]
        lenv *= psip[kk]
    end

    renv = ITensor(1)
    for kk = reverse(fA+1:LL)
        renv *= psi[kk]
        renv *= psip[kk]
    end

    mid = ITensor(1)
    for kk = iA:fA
        mid *= psi[kk]
        mid *= psip[kk]
    end

    t2 = lenv * mid'
    t2 *= lenv''
    t2 *= replaceprime(replaceprime(mid, 0=>3), 1=>0)
    t2 *= renv
    t2 *= renv''
    
    t2 = scalar(t2)

    return -log(t2)
end


""" For a folded tMPS, this is a cut at the end of the chain (middle of the TN)
but partial-tracing over all backward legs, leaving only the fw legs open in the RTM.
Memory cost here is chi^4, algorithm ~chi^5 at least """ 
function gen_renyi2_sym_openfwonly(psi::MPS, cut::Int)

    LL = length(psi)
    ss = siteinds(psi)

    psip = prime(linkinds, psi)
    psip2 = prime(linkinds, psi, 2)
    psip3 = prime(linkinds, psi, 3)

    lenv = ITensor(1)
    for kk = 1:cut-1
        lenv *= psi[kk]
        lenv *= psip[kk]
    end

    i1,i2,i3,i4 = Index.([2,2,2,2])

    renv = ITensor(1)
    for kk = reverse(cut:LL)
        renv *= reopen_ind(psi[kk], ss[kk], i4, i1)
        renv *= reopen_ind(psip[kk], ss[kk], i2, i1)
        renv *= reopen_ind(psip2[kk], ss[kk], i2, i3)
        renv *= reopen_ind(psip3[kk], ss[kk], i4, i3)

        @assert ndims(renv) < 5
    end


    t2 = lenv * renv
    t2 *= lenv''
    
    t2 = scalar(t2)

    return -log(t2)
end

