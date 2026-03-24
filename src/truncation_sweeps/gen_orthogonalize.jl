"""
Bring the MPS to generalized *left* canonical form without truncating (as far as possible)
This is symmetric, so we use symmetric eigenvalue decomposition, A => O D O^T with O complex orthogonal 
"""
function gen_canonical_left(in_mps::MPS)  # TODO: polar decomp?
    temp = deepcopy(in_mps)
    psi_leftgencan, _ = truncate_lsweep_sym(temp; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_leftgencan
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_rightgencan, _ = truncate_rsweep_sym(in_mps; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_rightgencan
end



""" Generalized canonical form to diagonalize symmetric RTM |psi^*><psi| 
bringing gen. orthogonality center in `ortho_center` """
function gen_canonical(in_psi::MPS, ortho_center::Int; cutoff::Float64=1e-13)

    mpslen = length(in_psi)
    #elt = eltype(in_psi[1])
    sits = siteinds(in_psi)
    sits_prime = prime(sits)
    maxdim = maxlinkdim(in_psi)

    # first bring to LEFT standard canonical form. 
    # Shouldn't matter if we are not truncating ... 
    psi_ortho = orthogonalize(in_psi, mpslen)

    #psi_ortho = copy(in_psi)

    XUinv= ITensors.OneITensor()
    right_env = ITensors.OneITensor()

    for ii = reverse(ortho_center+1:mpslen)
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        right_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        @assert order(right_env) == 2
        #@show tags(ind(right_env,1))
        F = symm_oeig(right_env, ind(right_env,1); cutoff, maxdim, tags=tags(ind(right_env,1)))
        U = F.V
        S = F.D
        #@show inds(U)

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U

        psi_ortho[ii] = Ai * XU

        right_env *= XU 
        right_env *= XU' 

    end

    # the last one 
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]


    XUinv= ITensors.OneITensor()
    left_env = ITensors.OneITensor()

    for ii = 1:ortho_center-1
        Ai = XUinv * psi_ortho[ii]

        left_env *= Ai
        left_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        @assert order(left_env) == 2
        F = symm_oeig(left_env, ind(left_env,1); cutoff, maxdim, tags=tags(ind(left_env,1)))
        U = F.V
        S = F.D

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U


        psi_ortho[ii] = Ai * XU

        left_env *= XU 
        left_env *= XU' 

    end
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]


    return noprime(linkinds, psi_ortho)

end

