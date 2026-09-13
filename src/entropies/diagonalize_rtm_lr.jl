"""
    rtm_eigvals(M::ITensor, l::Index)

Eigenvalues of the square two-index tensor `M` with indices `(l, l')`, the bond-space matrix
whose spectrum is that of the reduced transition matrix.

Without QNs this is a dense `geev`. `eigvals` (no eigenvectors) is the cheapest correct
route: computing the vectors as well costs 1.3-1.9x more for nothing, and going through
`ITensors.eigen` or [`ceigen`](@ref) adds the truncation and sorting this caller does not
want (`ceigen` refuses QNs outright, and promotes real input to complex).

With QNs `M` is block diagonal, so `ITensors.eigen` decomposes it sector by sector. Same
spectrum as densifying (machine precision, see `test_gen_renyi_entropies.jl`) but 1.2-3x
faster at χ = 256-512, and the noise floor of the non-normal RTM stays inside its own sector
instead of leaking across blocks.

No truncation keywords are passed on: the non-hermitian dense path in NDTensors does not
sort its eigenvalues, so the per-block truncation layered on top of it cannot be trusted
here. The caller applies the relative `cutoff` itself.
"""
function rtm_eigvals(M::ITensor, l::Index)
    hasqns(M) || return eigvals(M, (l, l'))
    return spectrum_vector(ITensors.eigen(M, l', l).D)
end


"""
    diagonalize_rtm_lr(psi, phi; normalize_eigs=true, cutoff=1e-10, sort_by_largest=true)

Eigenvalues of the reduced transition matrices `τ_k = Tr_{k+1…N} |phi⟩⟨psi|` at every cut
`k = 1 … N-1`

With `normalize_eigs=true` the eigenvalues are divided by `overlap_noconj(psi, phi) = Tr τ_k`
(the same at every cut), so they sum to 1 up to what `cutoff` removed.

Returns a length `N-1` vector of eigenvalue vectors, bond `1 … N-1`. The element type is
whatever `eigvals` returns: real inputs with an all-real spectrum give `Float64`
"""
function diagonalize_rtm_lr(psi::TMPSorMPS, phi::TMPSorMPS;
        normalize_eigs::Bool  = true,
        cutoff::Real          = 1e-10,
        sort_by_largest::Bool = true)
    psi, phi = unsided(psi), unsided(phi)  # accept a tagged boundary vector, work on the MPS

    N = length(psi)
    @assert length(phi) == N "psi and phi must have the same length ($(N) vs $(length(phi)))"

    psi = orthogonalize(noprime(psi), N)
    phi = orthogonalize(noprime(phi), N)
    phi = sim(linkinds, phi)
    match_siteinds!(psi, phi)

    # With QNs a bra cannot be contracted with a ket carrying the same arrows: reverse the
    # arrows of psi, data untouched (as in overlap_noconj). No-op without QNs.
    if arrows_clash(psi, phi)
        psi = transpose_arrows(psi)
    end

    EL = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env = (env * psi[j]) * phi[j]
        EL[j] = env
    end
    ov = scalar(env)

    ER = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in reverse(1:N)
        env = (env * psi[j]) * phi[j]
        ER[j] = env
    end

     #RTM is non-normal in general, so its small eigenvalues are sensitive: expect a noise floor 
     # relative to the largest. `cutoff` (relative to `max|λ|`) drops what lies below it.

    return map(1:N-1) do k
        lψ, lφ = linkind(psi, k), linkind(phi, k)
        l = dim(lψ) <= dim(lφ) ? lψ : lφ
        M = ER[k+1] * prime(EL[k], l)
        λ = rtm_eigvals(M, l)
        normalize_eigs && (λ = λ ./ ov)
        if cutoff > 0
            λmax = maximum(abs, λ)
            λ = filter(x -> abs(x) > cutoff * λmax, λ)
        end
        sort_by_largest ? sort(λ; by = abs, rev = true) : λ
    end
end


"""
    gen_renyi_entropies(psi, phi; cutoff=1e-10) -> (; S0, S05, S1, S2, S4)

Generalized Rényi entropies of the RTM `|phi⟩⟨psi|` at every cut, from its full eigenvalue
spectrum ([`diagonalize_rtm_lr`](@ref)). The non-symmetric counterpart of
[`gensym_renyi_entropies`](@ref), for any two MPS; `S2` agrees with [`gen_renyi2`](@ref),
which gets `Tr τ²` from the same environments by contraction.

Eigenvalues are normalized by `overlap_noconj(psi, phi)` and promoted to complex, so the
entropies are complex in general (principal branch of `log`: negative real eigenvalues give
`S1` an imaginary part even when `S2` is real). `S0` and `S05` count and weight the small
eigenvalues, so they depend on `cutoff`; `S1`, `S2`, `S4` do not in practice.
"""
function gen_renyi_entropies(psi::TMPSorMPS, phi::TMPSorMPS; cutoff::Real = 1e-10)
    eigs = diagonalize_rtm_lr(psi, phi; normalize_eigs = true, cutoff)
    return renyi_entropies([complex.(λ) for λ in eigs]; normalize_eigs = false)
end
