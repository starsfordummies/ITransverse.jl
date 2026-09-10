###############################################################################
#  Operations on sided boundary vectors
###############################################################################
#
# The type itself (and `TMPSorMPS` / `unsided`) lives in `ITenUtils/transverse_mps.jl`, first in
# that directory's include list because the utilities there already take a `TMPSorMPS`.
# Everything below needs those utilities in turn - `transpose_arrows`, `applyn`/`applyns`,
# `overlap_noconj` - so it is included after all of ITenUtils.

"""
    transpose(s::TransverseMPS)

The transposed boundary vector: the other side of the network, with the QN arrows reversed
and the data untouched (see [`transpose_arrows`](@ref)). This is the operation the symmetric
(RTM) algorithms perform implicitly when they use a right vector as its own bra.
"""
Base.transpose(s::TransverseMPS) = TransverseMPS(transpose_arrows(s.psi), flipside(s.side))



"""
    apply_column(O::MPO, s::TransverseMPS; kwargs...)

Apply a column tMPO to a boundary vector *from the correct side*: `applyn` for a right
vector, `applyns` for a left one. Returns a `TransverseMPS` on the same side.
"""
function apply_column(O::MPO, s::TransverseMPS; kwargs...)
    out = s.side === :right ? applyn(O, s.psi; kwargs...) : applyns(O, s.psi; kwargs...)
    return TransverseMPS(out, s.side)
end

""" Truncating version of [`apply_column`](@ref), returning `(TransverseMPS, singular values)`. """
function tapply_column(O::MPO, s::TransverseMPS; kwargs...)
    out, sv = s.side === :right ? tapply(O, s.psi; kwargs...) : tapplys(O, s.psi; kwargs...)
    return TransverseMPS(out, s.side), sv
end

# `tapply` does not name a direction, so on a boundary vector it can simply mean "apply this
# column from the side this vector is on" - which is `tapply_column`. Both the bare and the
# algorithm-first spellings, since callers use either.
tapply(O::MPO, s::TransverseMPS; kwargs...) = tapply_column(O, s; kwargs...)
tapply(alg, O::MPO, s::TransverseMPS; kwargs...) = tapply_column(O, s; alg, kwargs...)

# `applyn` / `applyns` / `applys` *do* name which legs they contract, so they stay the plain
# `MPS` primitives: picking the other one behind the caller's back would make the name lie.
# On a boundary vector the side already determines the answer - that is `apply_column`.
for (f, legs) in ((:applyn, "unprimed"), (:applyns, "primed"), (:applys, "primed"),
                  (:tapplys, "primed"))
    @eval function $f(::MPO, s::TransverseMPS; kwargs...)
        error("""
            `$($(QuoteNode(f)))` contracts the $($legs) legs, so it takes a plain `MPS`: on a
            :$(s.side) boundary vector it may or may not be the right one, and silently
            substituting the other would make the name mean nothing.
            Use `apply_column(O, psi)` (or `tapply_column`) to apply a column from the side
            the vector is on, or `$($(QuoteNode(f)))(O, unsided(psi))` to say you meant this one.""")
    end
end

"""
    overlap_noconj(l::TransverseMPS, r::TransverseMPS)

⟨l|r⟩ without conjugation, refusing to pair two vectors from the same side - that would be
an implicit transpose (use `transpose(l)` if you really mean it).
"""
function overlap_noconj(l::TransverseMPS, r::TransverseMPS; kwargs...)
    l.side === r.side && error("""
        Cannot pair two boundary vectors from the same side (:$(l.side)): contracting a
        vector with another one from the same side is an implicit transpose. Use
        `transpose(l)` to get the opposite-side vector explicitly.""")
    left, right = l.side === :left ? (l, r) : (r, l)
    return overlap_noconj(left.psi, right.psi; kwargs...)
end

# Mixed calls (one side tagged, one not) need no method of their own: `expval_LR` itself
# takes `TMPSorMPS` and unwraps. Only the both-tagged case below adds a check.

""" ⟨l|O|r⟩ with the sides checked, see [`overlap_noconj`](@ref). """
function expval_LR(l::TransverseMPS, O::MPO, r::TransverseMPS; kwargs...)
    l.side === r.side && error(
        "Cannot pair two boundary vectors from the same side (:$(l.side))")
    left, right = l.side === :left ? (l, r) : (r, l)
    return expval_LR(left.psi, O, right.psi; kwargs...)
end

###############################################################################
#  Tag-preserving returns
###############################################################################
#
# Return-strict: a routine that takes transverse boundary vectors hands back transverse
# boundary vectors. The side is never guessed - it is either the input's (a sweep, a
# re-gauging and a power method all leave a vector on the side it was on) or fixed by the
# function's own contract (`tlrapply` returns a left and a right, by construction). A
# column application preserves the chain length, so the boundary sites stay where they were
# and their counts ride along unchanged.

""" Re-wrap `out` on the side of `s`, or on `sd` when the function's contract fixes it. """
_retag(out::MPS, s::TransverseMPS, sd::Symbol = s.side) = TransverseMPS(out, sd)

# --- same side as the input ---------------------------------------------------
function truncate_sweep_sym(s::TransverseMPS; kwargs...)
    out, sv = truncate_sweep_sym(unsided(s); kwargs...)
    return _retag(out, s), sv
end

gen_canonical(s::TransverseMPS, ortho_center::Int; kwargs...) =
    _retag(gen_canonical(unsided(s), ortho_center; kwargs...), s)

function powermethod_sym(s::TransverseMPS, O::MPO, pm_params::PMParams; kwargs...)
    psi, info = powermethod_sym(unsided(s), O, pm_params; kwargs...)
    return _retag(psi, s), info
end

# --- sides fixed by the function's contract -----------------------------------
function powermethod_op(s::TransverseMPS; kwargs...)
    ll, rr, info = powermethod_op(unsided(s); kwargs...)
    return _retag(ll, s, :left), _retag(rr, s, :right), info
end

function powermethod_lr(s::TransverseMPS, L::MPO, R::MPO, pm_params::PMParams; kwargs...)
    ll, rr, info = powermethod_lr(unsided(s), L, R, pm_params; kwargs...)
    return _retag(ll, s, :left), _retag(rr, s, :right), info
end

# `TruncLR` is parametrised on its vector type, so it carries the tags rather than dropping
# them. The sides are the function's contract, not a guess: `L` is the left vector, `R` the
# right one, whatever the caller passed in.
_retag(t::TruncLR, l::TransverseMPS, r::TransverseMPS) =
    TruncLR(_retag(t.L, l, :left), _retag(t.R, r, :right), t.sv, t.ov_before, t.ov_after)

for f in (:tlapply, :trapply)
    @eval $f(l::TransverseMPS, A::MPO, r::TransverseMPS; kwargs...) =
        _retag($f(unsided(l), A, unsided(r); kwargs...), l, r)
end

tlrapply(l::TransverseMPS, AL::MPO, AR::MPO, r::TransverseMPS; kwargs...) =
    _retag(tlrapply(unsided(l), AL, AR, unsided(r); kwargs...), l, r)
