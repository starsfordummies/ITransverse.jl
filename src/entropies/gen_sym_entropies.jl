"""
Generalized entropy for a *symmetric* environment (psiL,psiL)
    Assuming we're in LEFT GENERALIZED canonical form, by default bring the MPS to it
    By default, normalizes the eigenvalues of the symmetric RTM. 
"""
function generalized_vn_entropy_symmetric(psiL::MPS; bring_gen_can::Bool=true, normalize_eigs::Bool=true)
    eigs_rtm = diagonalize_rtm_symmetric(psiL; bring_gen_can, normalize_eigs, sort_by_largest=false)
    return [salpha(eigs, 1) for eigs in eigs_rtm]
end


function generalized_r2_entropy_symmetric(psiL::MPS; bring_gen_can::Bool=true, normalize_eigs::Bool=true)
    eigs_rtm = diagonalize_rtm_symmetric(psiL; bring_gen_can, normalize_eigs, sort_by_largest=false)
    return [sum(eigs .^ 2) for eigs in eigs_rtm]
end


function generalized_svd_vn_entropy_symmetric(psi::MPS)
    _, svs = truncate_sweep_sym(psi; cutoff=1e-12, maxdim=maxlinkdim(psi), use_eig=false)
    return vn_from_matrix(svs)
end



""" We can compute the "SVD" VN entropy by just doing a right (generalized) sweep """
function generalized_svd_vn_entropy(psi::MPS, phi::MPS)
    truncp = (cutoff=1e-12, maxdim=maxlinkdim(psi)+maxlinkdim(phi), direction=:right)
    _, _, svs = truncate_sweep(psi, phi; truncp...)
    return vn_from_matrix(svs)
end


""" Given an input MPS `psi`, computes the symmetric generalized entropies by diagonalizing RTM
and builds alpha-order Renyi entropies as specified by `which_ents`. Returns a dict """
function gensym_renyi_entropies(psiL::MPS; which_ents=[0.5,1,2], bring_gen_can::Bool=true, normalize_eigs::Bool=true)
    eigs_rtm = diagonalize_rtm_symmetric(psiL; bring_gen_can, normalize_eigs, sort_by_largest=false)
    return renyi_entropies(eigs_rtm; which_ents, normalize_eigs=false)
end



""" Computes generalized entropies for a segment by diagonalizing the RTM - expensive!  (chi^4) """
function gensym_renyi_entropies_segment(psi::MPS, iA::Int, fA::Int; which_ents=[0.5,1,2], normalize_eigs::Bool=true)

    psig = ITransverse.gen_canonical(psi, iA+1)

    psigp = prime(linkinds, psig)
    rhoc = ITensor(1)
    for kk = iA:fA
        rhoc *= psig[kk]
        rhoc *= psigp[kk]
    end

    F = ITransverse.ITenUtils.symm_oeig(rhoc, (linkind(psig,iA-1), linkind(psig,fA)); cutoff=1e-13)
    eigvals_tau = normalize_eigs ? F.D/sum(F.D) : F.D


    renyi_entropies(eigvals_tau.tensor.storage.data; which_ents)
end


""" Computes the generalized SVD entropies: Given input MPS |phi> and <psi|, < diagonalizes the RTM |phi><psi| 
and builds alpha-order Renyi entropies as specified by `which_ents` (alpha=1 ie. VN is always computed). Returns a dict """
function gensvd_renyi_entropies(psi::MPS, phi::MPS; which_ents=[0.5,1,2], normalize_eigs::Bool=true)
 
    mpslen = length(psi)
    phi = sim(linkinds,phi)

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

    return renyi_entropies(svds_rdm; which_ents, normalize_eigs=false)

end


""" Slow compute generalized Renyi2 for symmetric case RTM (psi,psi) for an interval [iA-fA] """
function gen_renyi2_sym_interval_manual(psi::MPS, iA::Int, fA::Int)

    LL = length(psi)
    normalization = overlap_noconj(psi,psi)

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
    
    t2 = scalar(t2)/normalization^2

    return -log(t2)
end


""" For a folded tMPS, this is a cut at the end of the chain (middle of the TN)
but partial-tracing over all backward legs, leaving only the fw legs open in the RTM.
Memory cost here is chi^4, algorithm ~chi^5 at least """ 
function gen_renyi2_sym_openfwonly(psi::MPS, cut::Int)

    LL = length(psi)
    ss = siteinds(psi)

    normalization = overlap_noconj(psi,psi)

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
    
    t2 = scalar(t2)/normalization^2

    return -log(t2)
end
