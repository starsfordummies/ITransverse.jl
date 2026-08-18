"""
SVD kernel for the RTM truncation sweeps.

At every step of an RTM sweep the reduced transition matrix is never an
independent object: it is always the contraction of three pieces,

    rho[a, b] = sum_{i,j} R[a, i] * E[i, j] * L[b, j]

with `a` the open (site x kept-bond) index group of the R side, `b` the same on
the L side, and `i`, `j` the internal legs tying each outer factor to the
environment `E`. Hence

    rank(rho) <= min(dim(a), dim(b), dim(i), dim(j))

and whenever an internal leg is smaller than the open group of its own side,
building `rho` and taking a dense `dim(a) x dim(b)` SVD does work on a matrix
that is known to be rank deficient.

`svd_rtm` avoids that: each outer factor whose internal leg is strictly smaller
than its open group is QR'd first, the isometry is pushed out of the SVD, and
only the thin remainder is decomposed. This is exact -- the QR is a change of
basis inside the exactly representable row/column space of `rho`, so the
singular values are the singular values of `rho` and the singular vectors are
recovered by re-multiplying the isometries. It is the same manoeuvre as
QR-before-SVD in DMRG, and the array-based sweeps in `AllocEfficientTN`
(`truncated_svd_factored!`) already use the equivalent for their one-MPO path.
See also arXiv:2608.13805 Sec. II, which proposes the one-sided version.

Where it pays, for `D` = tMPS site dim, `d` = tMPO bond dim, `chi` = kept bond:

  * one tMPO (`trcontract`, i.e. `tlapply`/`trapply`): the bare-tMPS side has
    internal leg `chi` against an open group `D*chi`, so the deficit is a factor
    `D` and the saving is unconditional. Measured ~3x end to end on a folded
    Ising light-cone sweep at chi=96, and 3.0-4.7x per call for chi=64..384.
  * two tMPOs (`tlrcontract`): both sides carry a tMPO leg, `d*chi` against
    `D*chi`, so it pays only for `d < D`. The folded Ising columns have
    `D = d = 4`; there `svd_rtm` takes the dense route and costs nothing extra.

Pass `factored=false` to force the dense route (used by the tests to check the
two agree, and available as an escape hatch).
"""

"""
    svd_rtm(E, R, L, Ris; factored=true, kwargs...) -> (U, S, V, u)

Returns what `F = svd(E * R * L, Ris)` would give as `(F.U, F.S, F.V, F.u)`.

`factored` selects the route:

  * `true` (default) -- QR-compress the outer factors wherever that reduces the
    SVD *and* is measured to pay: dense tensors yes, block-sparse (QN) ones no
    (see the note at the branch below).
  * `false` -- always build `rho` and SVD it densely.
  * `:always` -- factorize whenever there is a rank deficit, QN included.

All three are exact and agree to machine precision; they differ only in cost.

`Ris` are the open indices of `R`, i.e. the ones the SVD keeps on the `U` side.
"""
function svd_rtm(E::ITensor, R::ITensor, L::ITensor, Ris;
        factored::Union{Bool, Symbol} = true,
        cutoff, maxdim, mindim, lefttags, righttags, kwargs...)

    Rint, Rop = commoninds(R, E), uniqueinds(R, E)
    Lint, Lop = commoninds(L, E), uniqueinds(L, E)

    # Same guard the explicit `rho` used to carry: rho has one (site) or two
    # (site + kept bond) indices per side, never more.
    @assert length(Rop) <= 2 && length(Lop) <= 2 "unexpected open inds: R $(Rop), L $(Lop)"

    function dense_route()
        rho = E * R * L
        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags, righttags, kwargs...)
        return F.U, F.S, F.V, F.u
    end

    factored === false && return dense_route()

    # `factored=true` means "factorize where it pays", and for block-sparse (QN)
    # tensors it does not. `qr` handles them correctly -- NDTensors implements it
    # per block and the result is exact to machine precision -- but the RTM SVD
    # is *already* block-diagonal there, so the sectors it decomposes are small
    # and the extra block-sparse contractions cost more than the smaller SVD
    # saves. Measured on the Z2 (SzParity) fw columns, D = d = 2, one tMPO,
    # N = 60: 0.70x at chi=128 and 0.82x at chi=256, against 1.81x / 1.67x for
    # the same shapes without QNs. `factored=:always` overrides -- worth trying
    # if a QN recipe ever has a site dimension well above its tMPO bond.
    if factored !== :always && any(hasqns, (E, R, L))
        return dense_route()
    end

    # Only factorize a side that is actually rank deficient; otherwise the QR is
    # pure overhead and the dense route is the right one.
    qrR = dim(Rint) < dim(Rop)
    qrL = dim(Lint) < dim(Lop)

    (qrR || qrL) || return dense_route()

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
