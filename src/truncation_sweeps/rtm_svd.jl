"""
SVD kernel for the RTM truncation sweeps.
The reduced transition matrix is typically the contraction of three pieces,

    rho[a, b] = sum_{i,j} R[a, i] * E[i, j] * L[b, j]

with `a` the open (site x kept-bond) index group of the R side, `b` the same on
the L side, and `i`, `j` the internal legs tying each outer factor to the
environment `E`. Hence

    rank(rho) <= min(dim(a), dim(b), dim(i), dim(j))

and whenever an internal leg is smaller than the open group of its own side,
building `rho` and taking a dense `dim(a) x dim(b)` SVD does work on a matrix
that is known to be rank deficient. `svd_ERL` avoids that by doing a QR before SVDing
"""

"""
    svd_ERL(E, R, L, Ris; factored=true, kwargs...) -> (U, S, V, u)

Returns what `F = svd(E * R * L, Ris)` would give as `(F.U, F.S, F.V, F.u)`.

`factored` selects the route:

  * `true` (default) - QR-compress the outer factors wherever that reduces the
    SVD and there are no QN (otherwise seems slower)
  * `false` - always build `rho` and SVD it densely.
  * `:always` - QR-factorize whenever there is a rank deficit, QN included.

`Ris` are the open indices of `R`, i.e. the ones the SVD keeps on the `U` side.
"""
function svd_ERL(E::ITensor, R::ITensor, L::ITensor, Ris;
        factored::Union{Bool, Symbol} = true,
        cutoff, maxdim, mindim, lefttags, righttags, kwargs...)

    Rint, Rop = commoninds(R, E), uniqueinds(R, E)
    Lint, Lop = commoninds(L, E), uniqueinds(L, E)

    # Index sanity
    @assert length(Rop) <= 2 && length(Lop) <= 2 "unexpected open inds: R $(Rop), L $(Lop)"

    function skipqr_svd()
        rho = E * R * L
        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags, righttags, kwargs...)
        return F.U, F.S, F.V, F.u
    end

    factored === false && return skipqr_svd()

    if factored !== :always && any(hasqns, (E, R, L))
        return skipqr_svd()
    end

    # Only factorize a side that is actually rank deficient
    qrR = dim(Rint) < dim(Rop)
    qrL = dim(Lint) < dim(Lop)

    (qrR || qrL) || return skipqr_svd()

    # `qr` handles QN (block-sparse) tensors too: NDTensors implements it
    # per-block, and the new link carries the QNs of the open group.
    Qr, Rr = qrR ? qr(R, Rop) : (nothing, R)
    Ql, Ll = qrL ? qr(L, Lop) : (nothing, L)

    # Row index of the SVD: the QR link if we factorized this side, else the
    # original open group.
    rowinds = qrR ? IndexSet(commonind(Qr, Rr)) : Ris

    T = (E * Rr) * Ll

    F = svd(T, rowinds; cutoff, maxdim, mindim, lefttags, righttags, kwargs...)

    # dag(Q)*Q = 1 on both sides, so re-multiplying keeps U, V isometric and
    # gives exactly the singular vectors of rho.
    U = qrR ? Qr * F.U : F.U
    V = qrL ? Ql * F.V : F.V

    return U, F.S, V, F.u
end
