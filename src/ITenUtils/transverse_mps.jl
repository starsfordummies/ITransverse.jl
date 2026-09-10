###############################################################################
#  Boundary vectors that know which side of the network they live on
###############################################################################

# In the transverse picture a boundary vector is either the *left* or the *right* edge of
# the column network, and the two are not interchangeable:
#
#   - a column tMPO is applied to a right vector with `applyn` and to a left vector with
#     `applyns` (they contract different legs),
#   - ⟨L|R⟩ pairs a left with a right vector; using the *same* vector on both sides is an
#     implicit transpose,
#   - with QNs that transpose reverses the arrows, so the mistake becomes a runtime error;
#     without QNs it is silent and simply gives the wrong number.
#
# `TransverseMPS` carries the side along with the state, so the builder's `LR` choice is no
# longer discarded the moment the vector is returned. Every transverse builder returns one:
# a plain `MPS` means "not a transverse boundary vector" (a real-space state, a folded
# sheet), and `unsided` is the deliberate way out.

"""
    TransverseMPS(psi::MPS, side::Symbol)

An MPS which knows the side of the network it belongs to (`:left` or `:right`).

Every transverse builder returns one - [`fw_tMPS`](@ref), [`folded_tMPS`](@ref),
[`fwback_tMPS`](@ref) - and [`sided`](@ref) wraps a bare state by hand. What it buys:

- [`apply_column`](@ref) picks `applyn` or `applyns` from the side, so a column can no
  longer be applied from the wrong side;
- `overlap_noconj` on two `TransverseMPS` refuses to pair two vectors from the same side;
- `transpose` flips the side *and* reverses the QN arrows ([`transpose_arrows`](@ref)),
  which is exactly the operation the "symmetric" algorithms perform implicitly when they
  reuse one vector as its own bra;
- `dag` flips the side too (conjugate *transpose*), while `conj` leaves it alone.

The side is the only metadata. A *non-product* boundary state is appended as its own site
and is then indistinguishable from a time site, but nothing here has to tell them apart:
the one place that cares - rebuilding an operator column over "the time sites of this
vector" - reads the bottom count off the blocks it is handed (`n_boundary_sites(b.rho0)`),
and the top of a vector on that path is always the product `vectorized_identity`, because
operators are inserted by the expval machinery rather than built into the vector. See
[`_time_sites`](@ref).

[`unsided`](@ref) (or `MPS(s)`) and `side(s)` get the parts back; the read-only MPS methods
are forwarded, and so are the transforms that leave the side alone (`orthogonalize`,
`normalize`, `replace_siteinds`) and the arithmetic (`psi * α`, `psi / α`, unary `-`, and
`+`/`-` between same-side vectors).

Passing one around needs no unwrapping at the call site: the routines that take a boundary
vector are typed [`TMPSorMPS`](@ref) and unwrap themselves, and those that *return* one
hand back a `TransverseMPS` (see the tag-preserving returns in `transverse_mps_ops.jl`).
The exception is deliberate: `applyn` / `applyns` / `applys` / `tapplys` name the legs they
contract, so they stay `MPS`-only and a tagged vector has to go through
[`apply_column`](@ref) / [`tapply_column`](@ref) - which is what makes applying a column
from the wrong side impossible rather than merely unlikely. `tapply` names no direction, so
on a tagged vector it *is* `tapply_column`.
"""
struct TransverseMPS
    psi::MPS
    side::Symbol
    function TransverseMPS(psi::MPS, side::Symbol)
        side in (:left, :right) ||
            error("TransverseMPS side must be :left or :right, got :$(side)")
        return new(psi, side)
    end
end

""" Wrap an MPS as a boundary vector on `side` (`:left` or `:right`), see
[`TransverseMPS`](@ref). [`unsided`](@ref) is the way back. """
sided(psi::MPS, side::Symbol) = TransverseMPS(psi, side)
sided(s::TransverseMPS, side::Symbol) = TransverseMPS(s.psi, side)

# Conversions 
ITensorMPS.MPS(s::TransverseMPS) = s.psi
Base.convert(::Type{MPS}, s::TransverseMPS) = s.psi

"""
    TMPSorMPS = Union{MPS, TransverseMPS}

A transverse boundary vector, with or without the [`TransverseMPS`](@ref) metadata.

This is the argument type of the routines that take a boundary vector but do not care
whether it is tagged - the sweeps, the entropies, the power method, the contraction
helpers. They unwrap with [`unsided`](@ref) and work on the plain `MPS`, so passing a
`TransverseMPS` is always allowed and never changes the result.

Deliberately *not* `Union{AbstractMPS, TransverseMPS}`: `MPO <: AbstractMPS`, so that
version would let an MPO through every slot that wants a state.

!!! note "A plain `MPS` in, a plain `MPS` out"
    The side is never invented. Hand these routines a `TransverseMPS` and you get one back
    (the side is either the input's or fixed by the function's own contract); hand them a
    bare `MPS` and you get a bare `MPS`, because there is nothing to carry.
"""
const TMPSorMPS = Union{MPS, TransverseMPS}

"""
    unsided(psi)

The plain `MPS` behind a boundary vector, tagged or not - the inverse of [`sided`](@ref) and
the unwrap-at-the-door helper for every routine that takes a [`TMPSorMPS`](@ref).

Total where `MPS(...)` is not: `MPS(psi)` converts a `TransverseMPS` but has no method for an
`MPS`, so `unsided` is what generic code should call.
"""
unsided(psi::AbstractMPS) = psi
unsided(s::TransverseMPS) = s.psi

""" Which side of the transverse network a [`TransverseMPS`](@ref) lives on. """
side(s::TransverseMPS) = s.side


# read-only forwarding, so a TransverseMPS can be inspected like an MPS
Base.length(s::TransverseMPS) = length(s.psi)
Base.getindex(s::TransverseMPS, i...) = getindex(s.psi, i...)
Base.eachindex(s::TransverseMPS) = eachindex(s.psi)
Base.copy(s::TransverseMPS) = TransverseMPS(copy(s.psi), s.side)
ITensorMPS.siteinds(s::TransverseMPS) = siteinds(s.psi)
ITensorMPS.linkinds(s::TransverseMPS) = linkinds(s.psi)
ITensorMPS.linkdim(s::TransverseMPS, b::Integer) = linkdim(s.psi, b)
ITensorMPS.linkdims(s::TransverseMPS) = linkdims(s.psi)
ITensorMPS.maxlinkdim(s::TransverseMPS) = maxlinkdim(s.psi)
ITensors.hasqns(s::TransverseMPS) = hasqns(s.psi)
LinearAlgebra.norm(s::TransverseMPS) = norm(s.psi)

# Tag-preserving transforms: re-gauging and normalising change neither the side nor the
# boundary-site counts, so the wrapper travels with the result.
ITensorMPS.orthogonalize(s::TransverseMPS, j::Int; kw...) =
    TransverseMPS(orthogonalize(s.psi, j; kw...), s.side)
ITensorMPS.orthogonalize!(s::TransverseMPS, j::Int; kw...) = (orthogonalize!(s.psi, j; kw...); s)
LinearAlgebra.normalize(s::TransverseMPS) = TransverseMPS(normalize(s.psi), s.side)
LinearAlgebra.normalize!(s::TransverseMPS) = (normalize!(s.psi); s)
ITensorMPS.ortho_lims(s::TransverseMPS) = ortho_lims(s.psi)
ITensorMPS.replace_siteinds(s::TransverseMPS, sites) =
    TransverseMPS(replace_siteinds(s.psi, sites), s.side)
ITensorMPS.replace_siteinds!(s::TransverseMPS, sites) = (replace_siteinds!(s.psi, sites); s)
ITensorMPS.truncate(s::TransverseMPS; kw...) = TransverseMPS(truncate(s.psi; kw...), s.side)
ITensorMPS.truncate!(s::TransverseMPS; kw...) = (truncate!(s.psi; kw...); s)

# Arithmetic: rescaling by a scalar changes neither the side nor which site is the
# orthogonality center, so it forwards to the same `_apply_to_orthocenter` machinery that
# backs `psi * α` / `psi / α` for a plain MPS.
Base.:*(s::TransverseMPS, α::Number) = TransverseMPS(s.psi * α, s.side)
Base.:*(α::Number, s::TransverseMPS) = TransverseMPS(α * s.psi, s.side)
Base.:/(s::TransverseMPS, α::Number) = TransverseMPS(s.psi / α, s.side)
Base.:-(s::TransverseMPS) = TransverseMPS(-s.psi, s.side)

"""
    +(s⃗::TransverseMPS...; kwargs...)
    -(s::TransverseMPS, t::TransverseMPS; kwargs...)

Linear combination of boundary vectors, forwarding to the `AbstractMPS` `+`/`-` (density-
matrix sum, or pass `alg = "directsum"`). All operands must be on the same side - mixing
`:left` and `:right` here is the same mistake [`overlap_noconj`](@ref) already refuses - and
the result carries that common side.
"""
function Base.:+(s⃗::TransverseMPS...; kwargs...)
    sd = s⃗[1].side
    all(s -> s.side === sd, s⃗) ||
        error("cannot add TransverseMPS on different sides: $(unique(side.(s⃗)))")
    return TransverseMPS(+(map(s -> s.psi, s⃗)...; kwargs...), sd)
end
function Base.:-(s::TransverseMPS, t::TransverseMPS; kwargs...)
    s.side === t.side ||
        error("cannot subtract TransverseMPS on different sides: :$(s.side) and :$(t.side)")
    return TransverseMPS(-(s.psi, t.psi; kwargs...), s.side)
end

"""
    dag(s::TransverseMPS)

Conjugate transpose. `dag` **flips the side**: the dag of a `:left` vector is a `:right`
one. That is the meaning of the operation, not an artefact of the QN arrows - it holds with
or without symmetry, so the tag never depends on `hasqns`.

Use [`conj`](@ref) to conjugate *without* transposing (side unchanged), or
[`transpose`](@ref) to transpose without conjugating.
"""
function ITensors.dag(s::TransverseMPS)
    # once per session: the side change is correct but easy to miss at a call site
    @warn "dag(::TransverseMPS) is a conjugate *transpose*: it turns a :$(s.side) vector \
           into a :$(flipside(s.side)) one. `conj` conjugates without flipping the side." maxlog=1
    return TransverseMPS(dag(s.psi), flipside(s.side))
end

""" Conjugate the data, leaving the side alone - the non-transposing half of [`dag`](@ref). """
Base.conj(s::TransverseMPS) = TransverseMPS(conj(s.psi), s.side)

function Base.show(io::IO, s::TransverseMPS)
    println(io, "TransverseMPS[$(s.side)] of length $(length(s.psi)), χ=$(maxlinkdim(s.psi))",
            hasqns(s.psi) ? " (QN)" : "")
end

""" The opposite side. """
flipside(sd::Symbol) = sd === :left ? :right : :left
