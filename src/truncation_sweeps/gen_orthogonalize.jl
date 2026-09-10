""" Single directional sweep of the generalized canonicalization.
Iterates over `sweep_range`, updates `psi[ii]` in place, and returns
the `XUinv` that must be absorbed into the next site. """
function _gen_canonical_sweep!(psi::MPS, sweep_range, sits, sits_prime; cutoff, maxdim)
    XUinv = ITensors.OneITensor()
    env   = ITensors.OneITensor()
    for ii in sweep_range
        Ai = XUinv * psi[ii]

        env *= Ai
        # bra copy of Ai: arrows reversed (QNs only), data *not* conjugated. `dag` on the
        # site index keeps it contractible with Ai's; both are no-ops without QNs.
        env *= replaceind(transpose_arrows(Ai)', sits_prime[ii] => dag(sits[ii]))

        @assert order(env) == 2
        F = symm_oeig(env, ind(env, 1); cutoff, maxdim, lefttags=tags(ind(env, 1)))
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
function gen_canonical(in_psi::TMPSorMPS, ortho_center::Int; cutoff::Float64=1e-13)
    in_psi = unsided(in_psi)  # accept a tagged boundary vector, work on the MPS

    no_qns_supported("gen_canonical (generalized canonical form)", in_psi;
        hint="It relies on `symm_oeig`. Entropies can still be computed with `bring_gen_can=false`.")

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

