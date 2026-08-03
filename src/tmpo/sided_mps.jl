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
# `SidedMPS` carries the side along with the state, so the builder's `LR` choice is no
# longer discarded the moment the vector is returned. It is entirely opt-in: every builder
# still returns a plain `MPS` by default.

"""
    SidedMPS(psi::MPS, side::Symbol)

A transverse boundary vector tagged with the side of the network it belongs to
(`:left` or `:right`).

Wrapping is opt-in - pass `sided=true` to [`fw_tMPS`](@ref) / [`folded_tMPS`](@ref), or wrap
by hand with [`sided`](@ref). The wrapper buys three things:

- [`apply_column`](@ref) picks `applyn` or `applyns` from the side, so a column can no
  longer be applied from the wrong side;
- `overlap_noconj` on two `SidedMPS` refuses to pair two vectors from the same side;
- `transpose` flips the side *and* reverses the QN arrows ([`transpose_arrows`](@ref)),
  which is exactly the operation the "symmetric" algorithms perform implicitly when they
  reuse one vector as its own bra.

`MPS(s)` and `side(s)` get the parts back; most read-only MPS methods are forwarded.
"""
struct SidedMPS
    psi::MPS
    side::Symbol
    function SidedMPS(psi::MPS, side::Symbol)
        side in (:left, :right) ||
            error("SidedMPS side must be :left or :right, got :$(side)")
        return new(psi, side)
    end
end

""" Wrap an MPS as a boundary vector on `side` (`:left` or `:right`). """
sided(psi::MPS, side::Symbol) = SidedMPS(psi, side)
sided(s::SidedMPS, side::Symbol) = SidedMPS(s.psi, side)

""" The underlying MPS of a [`SidedMPS`](@ref), dropping the side tag.

Spelled as a conversion, so anything in the package that still takes a plain `MPS`
(`powermethod_sym`, `truncate_sweep`, `gensym_renyi_entropies`, `orthogonalize`, ...) is one
`MPS(...)` away. """
ITensorMPS.MPS(s::SidedMPS) = s.psi
Base.convert(::Type{MPS}, s::SidedMPS) = s.psi

""" Which side of the transverse network a [`SidedMPS`](@ref) lives on. """
side(s::SidedMPS) = s.side

# read-only forwarding, so a SidedMPS can be inspected like an MPS
Base.length(s::SidedMPS) = length(s.psi)
Base.getindex(s::SidedMPS, i...) = getindex(s.psi, i...)
Base.eachindex(s::SidedMPS) = eachindex(s.psi)
Base.copy(s::SidedMPS) = SidedMPS(copy(s.psi), s.side)
ITensorMPS.siteinds(s::SidedMPS) = siteinds(s.psi)
ITensorMPS.linkinds(s::SidedMPS) = linkinds(s.psi)
ITensorMPS.maxlinkdim(s::SidedMPS) = maxlinkdim(s.psi)
ITensors.hasqns(s::SidedMPS) = hasqns(s.psi)
LinearAlgebra.norm(s::SidedMPS) = norm(s.psi)

function Base.show(io::IO, s::SidedMPS)
    println(io, "SidedMPS[$(s.side)] of length $(length(s.psi)), χ=$(maxlinkdim(s.psi))",
            hasqns(s.psi) ? " (QN)" : "")
end

""" The opposite side. """
flipside(sd::Symbol) = sd === :left ? :right : :left

"""
    transpose(s::SidedMPS)

The transposed boundary vector: the other side of the network, with the QN arrows reversed
and the data untouched (see [`transpose_arrows`](@ref)). This is the operation the symmetric
(RTM) algorithms perform implicitly when they use a right vector as its own bra.
"""
Base.transpose(s::SidedMPS) = SidedMPS(transpose_arrows(s.psi), flipside(s.side))

"""
    apply_column(O::MPO, s::SidedMPS; kwargs...)

Apply a column tMPO to a boundary vector *from the correct side*: `applyn` for a right
vector, `applyns` for a left one. Returns a `SidedMPS` on the same side.
"""
function apply_column(O::MPO, s::SidedMPS; kwargs...)
    out = s.side === :right ? applyn(O, s.psi; kwargs...) : applyns(O, s.psi; kwargs...)
    return SidedMPS(out, s.side)
end

""" Truncating version of [`apply_column`](@ref), returning `(SidedMPS, singular values)`. """
function tapply_column(O::MPO, s::SidedMPS; kwargs...)
    out, sv = s.side === :right ? tapply(O, s.psi; kwargs...) : tapplys(O, s.psi; kwargs...)
    return SidedMPS(out, s.side), sv
end

"""
    overlap_noconj(l::SidedMPS, r::SidedMPS)

⟨l|r⟩ without conjugation, refusing to pair two vectors from the same side - that would be
an implicit transpose (use `transpose(l)` if you really mean it).
"""
function overlap_noconj(l::SidedMPS, r::SidedMPS; kwargs...)
    l.side === r.side && error("""
        Cannot pair two boundary vectors from the same side (:$(l.side)): contracting a
        vector with another one from the same side is an implicit transpose. Use
        `transpose(l)` to get the opposite-side vector explicitly.""")
    left, right = l.side === :left ? (l, r) : (r, l)
    return overlap_noconj(left.psi, right.psi; kwargs...)
end

""" ⟨l|O|r⟩ with the sides checked, see [`overlap_noconj`](@ref). """
function expval_LR(l::SidedMPS, O::MPO, r::SidedMPS; kwargs...)
    l.side === r.side && error(
        "Cannot pair two boundary vectors from the same side (:$(l.side))")
    left, right = l.side === :left ? (l, r) : (r, l)
    return expval_LR(left.psi, O, right.psi; kwargs...)
end
