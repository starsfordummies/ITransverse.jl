"""
Diagonalize the symmetric RTM |psi*><psi| by sweeping from one end.

    `   
    |  |  |  |  |   _____                ____
    D--D--D--D--D--|     |   (same λ) --|    |
                   | renv|      ≃       |    |
    D--D--D--D--D--|_____|            --|____|
    |  |  |  |  |

    `

- `direction = :right` sweeps N→2, building right environments
  (gen. right canonical: ortho center at site N)
- `direction = :left`  sweeps 1→N-1, building left environments
  (gen. right canonical form: ortho center at site 1)

Returns a length N-1 vector of eigenvalue vectors, ordered bond 1 … N-1.
"""
function diagonalize_rtm_symmetric(psi::MPS;
    direction::Symbol        = :right,
    bring_gen_can::Bool      = true,
    normalize_eigs::Bool     = true,
    sort_by_largest::Bool    = true,
    cutoff::Float64          = 1e-12)

    mpslen = length(psi)

    if bring_gen_can
        ortho_center = direction == :left ? 1 : mpslen
        psi = gen_canonical(psi, ortho_center)
        psi[end] /= sqrt(overlap_noconj(psi, psi))
    end

    overlap_val = overlap_noconj(psi, psi)
    if abs(1 - overlap_val) > 1e-4 && !normalize_eigs
        @warn "overlap not 1: $(overlap_val)"
    end

    sweep = direction == :left ? (1:mpslen-1) : (mpslen:-1:2)
    psiP  = prime(linkinds, psi)
    env   = ITensors.OneITensor()

    eigenvalues_rtm = Vector{Vector}(undef, mpslen - 1)

    for ii in sweep
        env *= psi[ii]
        env *= psiP[ii]
        @assert order(env) == 2 "unexpected env order at site $ii: $(inds(env))"

        eigss = eigvals(env)

        if normalize_eigs
            eigss = eigss ./ sum(eigss)
        elseif abs(sum(eigss) - 1.0) > 0.01
            bond = direction == :left ? ii : ii - 1
            @warn "RTM not well normalized at bond $bond: Σeigs-1 = $(abs(sum(eigss) - 1.0))"
        end

        if sort_by_largest
            eigss = sort(filter(x -> abs(x) >= cutoff, eigss), by=abs, rev=true)
        end

        eigenvalues_rtm[direction == :left ? ii : ii - 1] = eigss
    end

    return eigenvalues_rtm
end

# Thin wrapper kept for backward compatibility
diagonalize_rtm_right_gen_sym(psi::MPS; bring_right_gen::Bool=false, kwargs...) =
    diagonalize_rtm_symmetric(psi; direction=:left, bring_gen_can=bring_right_gen, kwargs...)


""" Alt recipe for diagonalizing symmetric RTM, maybe more stable but slower """
function diagonalize_rtm_symmetric_alt(psi::MPS; 
    direction::Symbol=:right,
    maxdim = maxlinkdim(psi),
    kwargs...)

    ss = siteinds(psi)
    N = length(ss)
    psip = prime(linkinds, psi)

    sweep_start, sweep_step, sweep_end, sv_offset = if direction == :right
        N, -1, 2, 0
    elseif direction == :left
        1, 1, N-1, -1
    else
        error("direction must be :left or :right")
    end

    last_site = sweep_end + sweep_step  # :right → 1, :left → N

    eigen_all = zeros(ComplexF64, N-1, maxdim)
    envs = Vector{ITensor}(undef, N)

    # Build environments sweeping away from last_site
    env = ITensors.OneITensor()
    for ii = sweep_start:sweep_step:sweep_end
        env *= psi[ii]
        env *= psip[ii]
        envs[ii] = env  # :right → envs[N]..envs[2], :left → envs[1]..envs[N-1]
    end

    # First step: last_site contracts with the innermost environment
    work = psi[last_site]

    rho = work * envs[sweep_end]  # :right → envs[2], :left → envs[N-1]
    rho *= work'
    @assert order(rho) == 2 "check your inds? $(inds(rho))"

    F = symm_oeig(rho, ss[last_site], ss[last_site]'; maxdim, kwargs...)
    work *= F.V  # no dag, it's orthogonal

    Dvec = Array(storage(F.D).data)/sum(F.D)

    eigen_all[last_site + sv_offset, 1:length(Dvec)] .= Dvec  # :right → bond 1, :left → bond N-1

    for jj = sweep_end:-sweep_step:sweep_start+sweep_step
        work *= psi[jj]

        rho = work * envs[jj - sweep_step]
        rho *= work'
        @assert order(rho) == 4 "check your inds? $(inds(rho))"

        F = symm_oeig(rho, (ss[jj], F.l), (ss[jj]', F.l'); maxdim, kwargs...)
        work *= F.V

        Dvec = Array(storage(F.D).data)/sum(F.D)
        eigen_all[jj + sv_offset, 1:length(Dvec)] .= Dvec
    end

    return eigen_all
end
