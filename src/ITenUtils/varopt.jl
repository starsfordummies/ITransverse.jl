""" Attempts at variational optimizers """

using ITensors, ITensorMPS, ITransverse 

function build_renvs(psi::MPS, phi::MPS)
    LL = length(psi)
    right_envs = [ITensor(1) for _ = 1:LL+1]
    for jj = reverse(2:LL)
        right_envs[jj] = (right_envs[jj+1] * psi[jj]) * dag(phi[jj])
    end
    return right_envs
end

function varopt(target::MPS; guess=nothing, nsweeps::Int=1, verbose::Bool=false, kwargs...)

#truncp = TruncParams()
#nsweeps = 5

# ss = siteinds("S=1/2", 40)
# target = random_mps(ss, linkdims=60)
# target = add(target,target,mindim=100)

    ss = siteinds(target)
    work = something(guess, random_mps(ss, linkdims=2))
    LL = length(target)

    orthogonalize!(work,1)
    right_envs = build_renvs(target, work)
    left_envs = [ITensor(1) for _ = 1:LL+1]

    @assert length(target) == LL

    links = linkinds(work)
    #F = ITensors.TruncSVD()
    ss = siteinds(work)

    for nn = 1:nsweeps
        li = ss[1]

        for ii = 1:LL-1
            bij = left_envs[ii] * target[ii]
            bij *= target[ii+1]
            bij *= right_envs[ii+2]
            # @show ii
            # @show li 
            # @show inds(bij)
            F = svd(bij, li; kwargs...) #, max_dim=truncp.chimax)
            # @show inds(F.U)
            work[ii] = F.U 
            li = Index(dim(F.u), "Link,l=$(ii)")
            work[ii] *= delta(F.u, li)
            work[ii+1] = (F.S * F.V) * delta(F.u, li)
            li = [li, ss[ii+1]]
            # update left env 
            left_envs[ii+1] = (left_envs[ii] * target[ii])* dag(work[ii])
            #work[ii+1] = (s * v) * work[ii+1]
        end

        ri = ss[LL]
        for ii = reverse(2:LL)
            bij = right_envs[ii+1] * target[ii]
            bij *= target[ii-1]
            bij *= left_envs[ii-1]
            # @show ii
            # @show ri
            # @show inds(bij)
            F = svd(bij, ri; kwargs...)
            # @show inds(F.U)
            work[ii] = F.U 
            ri = Index(dim(F.u), "Link,l=$(ii-1)")
            work[ii] *= delta(F.u, ri)
            work[ii-1] = (F.S * F.V) * delta(F.u, ri)
            ri = [ss[ii-1], ri]
            # update right env 
            right_envs[ii] = (right_envs[ii+1] * target[ii])* dag(work[ii])
            #work[ii+1] = (s * v) * work[ii+1]
        end
        verbose && @info nn, inner(work, target), maxlinkdim(work), maxlinkdim(target)
    end

    return work 
end

"""
    apply_variational(Ut::MPO, psi::MPS; cutoff, maxdim, nsweeps=2, expand_cutoff=1e-14, verbose=false)

Apply `Ut` to `psi` using exact (untruncated) expansion followed by
variational (ALS-style, two-sweep-per-iteration) recompression to bond
dimension `maxdim` via [`varopt`](@ref), instead of the single-pass
SVD-truncated `apply`. This tends to give a better rank-`maxdim`
approximation of `Ut*psi` than naive truncation, at the cost of extra
sweeps. `psi` is used as the initial guess for the compressed state, since
for a single time step `Ut*psi` should stay close to `psi`.

`expand_cutoff` controls the (near-)exact expansion step; it should be much
tighter than `cutoff`/`maxdim` (which control the final compression).
"""
function apply_variational(Ut::MPO, psi::MPS; cutoff::Real, maxdim::Int,
        nsweeps::Int=2, expand_cutoff::Real=1e-14, verbose::Bool=false)
    target = apply(Ut, psi; cutoff=expand_cutoff, maxdim=typemax(Int))
    guess = copy(psi)
    return varopt(target; guess, nsweeps, cutoff, maxdim, verbose)
end
