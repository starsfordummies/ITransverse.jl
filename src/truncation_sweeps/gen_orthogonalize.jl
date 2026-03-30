""" Single directional sweep of the generalized canonicalization.
Iterates over `sweep_range`, updates `psi[ii]` in place, and returns
the `XUinv` that must be absorbed into the next site. """
function _gen_canonical_sweep!(psi::MPS, sweep_range, sits, sits_prime; cutoff, maxdim)
    XUinv = ITensors.OneITensor()
    env   = ITensors.OneITensor()
    for ii in sweep_range
        Ai = XUinv * psi[ii]

        env *= Ai
        env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        @assert order(env) == 2
        F = symm_oeig(env, ind(env, 1); cutoff, maxdim, tags=tags(ind(env, 1)))
        U, S = F.V, F.D

        XU    = U * S .^ -0.5
        XUinv = S .^ 0.5 * U

        psi[ii] = Ai * XU

        env *= XU
        env *= XU'
    end
    return XUinv
end

""" Generalized canonical form to diagonalize symmetric RTM |psi^*><psi| 
bringing gen. orthogonality center in `ortho_center` """
function gen_canonical(in_psi::MPS, ortho_center::Int; cutoff::Float64=1e-13)

    mpslen  = length(in_psi)
    sits    = siteinds(in_psi)
    sits_prime = prime(sits)
    maxdim  = maxlinkdim(in_psi)

    # first bring to standard canonical form
    psi_ortho = orthogonalize(in_psi, 1)

    XUinv = _gen_canonical_sweep!(psi_ortho, 1:ortho_center-1, sits, sits_prime; cutoff, maxdim)
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]

    XUinv = _gen_canonical_sweep!(psi_ortho, reverse(ortho_center+1:mpslen), sits, sits_prime; cutoff, maxdim)
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]


    return noprime(linkinds, psi_ortho)

end

