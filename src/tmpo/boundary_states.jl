###############################################################################
#  Boundary (initial/final) states for the transverse builders
###############################################################################

# # Boundary states
#
# Every transverse builder (`fw_tMPO`, `fw_tMPS`, `fwback_tMPO`, `fwback_tMPS`,
# `folded_tMPO`, `folded_tMPS`, `folded_tMPO_ext`) closes the temporal chain with a
# *boundary state*: `bl` at the bottom (the initial state) and `tr`/`fold_op` at the
# top (final state or operator).
#
# A boundary state is **one column** of the space-like boundary MPS:
#
# ```
#  ⟨ϕ_f|      B———B———B———B———B      <- top    boundary (`tr` / `fold_op`)
#              |   |   |   |   |
#  U           W———W———W———W———W
#              |   |   |   |   |
#  |ψ_0⟩       A———A———A———A———A      <- bottom boundary (`bl`)
#
#             ⟨L|  T   T   T  |R⟩
# ```
#
# * **Product states** are rank-1 (`Vector` or 1-leg `ITensor`). They are *contracted*
#   into the first/last tensor of the temporal chain, which keeps `Nt` sites.
# * **Non-product states** carry the bond indices of the boundary MPS. Those bonds run
#   along the *spatial* direction, so they become **site** indices of the temporal
#   chain: the boundary tensor is *appended as an extra site*. A tMPO with a
#   non-product `bl` has `Nt+1` sites, with non-product `bl` *and* `tr` it has `Nt+2`.
#   - bulk columns (tMPO): rank-3 `(phys, left, right)`
#   - edge columns (tMPS): rank-2 `(phys, bond)`
#
# ## Index convention
#
# 1. The leg contracted into the temporal chain (the *physical/space* leg of the
#    boundary MPS) is the one tagged `"Site"`, and it must be the only one.
# 2. The spatial bonds are a **prime pair** `(s, s')` of a single index `s`:
#    - `s` (unprimed) points **right**: it is the leg an MPO column is contracted
#      *from* (cf. `applyn(T, ψR)`, which contracts unprimed legs),
#    - `s'` (primed) points **left**: the output leg.
#    All columns therefore share the same boundary bond index (and bond dimension).
# 3. Rank-2 (edge) tensors keep a single *unprimed* leg `s`, whichever side it is on,
#    because it plays the role of a site index of the edge tMPS.
#
# Build tensors in this convention with [`boundary_tensor`](@ref), get the two edge
# tensors of a translation-invariant boundary MPS with [`close_boundary`](@ref), and
# fold them (`A ⊗ conj(A)`) with [`fold_boundary`](@ref).
#
# Typical use, for a boundary MPS with uniform bulk tensor `A` and edge vectors `vL`, `vR`:
#
# ```julia
# bl  = boundary_tensor(A; phys=σ, left=l, right=r)   # rank-3, bulk column
# blL = close_boundary(bl, vL; side=:left)            # rank-2, leftmost column
# blR = close_boundary(bl, vR; side=:right)           # rank-2, rightmost column
#
# tp = tMPOParams(mp; dt, init_state=bl)
# b  = FwtMPOBlocks(tp)
#
# T  = fw_tMPO(b, ts; tr=up_state)                     # Nt+1 sites
# L  = fw_tMPS(b, ts; LR=:left,  bl=blL, tr=up_state)  # Nt+1 sites
# R  = fw_tMPS(b, ts; LR=:right, bl=blR, tr=up_state)
# ```
#

"""
    to_boundary(x; tags="Site,bl")

Normalize a user-supplied boundary state (`Vector` or `ITensor`) into an `ITensor` obeying
the boundary convention: the leg contracted into the temporal chain carries `tags`
(which must include `"Site"`), the others are left untouched.
"""
to_boundary(x::AbstractVector; tags="Site,bl") = ITensor(complex(collect(x)), Index(length(x), tags))
function to_boundary(x::ITensor; tags="Site,bl")
    ndims(x) == 1 && return settags(x, tags)
    return check_boundary(x)
end

""" Index of a boundary tensor which gets contracted *into* the temporal chain. """
boundary_phys_ind(A::ITensor) = ndims(A) == 1 ? ind(A, 1) : only(inds(A, "Site"))

""" Spatial bond indices of a boundary tensor (empty tuple for product states). """
boundary_bond_inds(A::ITensor) = uniqueinds(inds(A), boundary_phys_ind(A))

""" The (unprimed) spatial bond index of a boundary tensor, `nothing` for product states. """
function boundary_bond_ind(A::ITensor)
    bb = boundary_bond_inds(A)
    isempty(bb) && return nothing
    return noprime(first(bb))
end

""" `true` if the boundary state is a product state (rank-1), ie. does not add sites. """
is_product_boundary(A::ITensor) = ndims(A) == 1
is_product_boundary(::AbstractVector) = true

""" Number of extra sites a boundary state adds to a temporal chain (0 or 1). """
n_boundary_sites(A) = is_product_boundary(A) ? 0 : 1

""" Bond dimension of a boundary state (1 for product states). """
boundary_linkdim(A::ITensor) = is_product_boundary(A) ? 1 : dim(boundary_bond_ind(A))

"""
    check_boundary(A::ITensor)

Check that `A` obeys the boundary-state convention (see [`boundary_tensor`](@ref)):
rank 1 (product), rank 2 (edge column, one unprimed bond) or rank 3
(bulk column, bonds `(s, s')`). Throws a descriptive error otherwise.
"""
function check_boundary(A::ITensor)
    nd = ndims(A)
    nd == 1 && return A

    length(inds(A, "Site")) == 1 || error("""
        Boundary tensor with $(nd) legs must have exactly one leg tagged "Site" \
        (the one contracted into the temporal chain), got inds $(inds(A)).
        Build it with `boundary_tensor(A; phys, left, right)`.""")

    bonds = boundary_bond_inds(A)

    if nd == 2
        plev(only(bonds)) == 0 || error(
            "The bond leg of a rank-2 (edge) boundary tensor must be unprimed, got $(only(bonds))")
    elseif nd == 3
        s1, s2 = bonds
        (noprime(s1) == noprime(s2) && plev(s1) + plev(s2) == 1) || error("""
            The two bond legs of a rank-3 (bulk) boundary tensor must be a prime pair (s, s'),
            got $(s1) and $(s2). Build it with `boundary_tensor(A; phys, left, right)`.""")
    else
        error("Boundary states must have 1, 2 or 3 legs, got $(nd): $(inds(A))")
    end
    return A
end

"""
    boundary_tensor(A::ITensor; phys::Index, left=nothing, right=nothing, bond_ind=nothing)

Put the boundary-MPS tensor `A` in the ITransverse boundary convention:
`phys` (the leg contracted into the temporal chain) is tagged `"Site"`, the `right`
bond becomes the unprimed index `s` and the `left` bond becomes `s'`.

If only one of `left`/`right` is given (edge column of the boundary MPS) the
surviving bond is left *unprimed*.

Pass `bond_ind=s` to reuse the boundary bond index of another column - all columns
of the same boundary MPS must share it. See also [`close_boundary`](@ref).
"""
function boundary_tensor(A::ITensor; phys::Index, left=nothing, right=nothing,
                         bond_ind::Union{Index,Nothing}=nothing, tags="bdry")

    hasind(A, phys) || error("Tensor does not have the physical index $(phys): $(inds(A))")

    bonds = filter(!isnothing, (left, right))
    out = settags(A, "Site", phys)

    isempty(bonds) && return check_boundary(out)

    allequal(dim.(bonds)) || error("Boundary bonds must have equal dimensions, got $(dim.(bonds))")
    s = something(bond_ind, Index(dim(first(bonds)), tags))
    dim(s) == dim(first(bonds)) ||
        error("bond_ind has dimension $(dim(s)) but the boundary bonds have $(dim(first(bonds)))")

    if length(bonds) == 1  # edge column: single unprimed leg
        out = replaceind(out, only(bonds) => s)
    else
        out = replaceinds(out, (right, left) => (s, s'))
    end

    return check_boundary(out)
end

boundary_tensor(A::AbstractVector; kwargs...) = to_boundary(A)

"""
    close_boundary(A::ITensor, v; side::Symbol)

Close the `side` (`:left` or `:right`) bond of a rank-3 (bulk) boundary tensor with the
vector `v`, returning the rank-2 tensor of the corresponding *edge* column of the
boundary MPS. The surviving leg keeps the (unprimed) boundary bond index of `A`, so the
edge tMPS and the bulk tMPO automatically share their extra site index.
"""
function close_boundary(A::ITensor, v; side::Symbol)
    ndims(A) == 3 || error("close_boundary expects a rank-3 (bulk) boundary tensor, got $(ndims(A)) legs")
    check_boundary(A)
    s = boundary_bond_ind(A)
    vt = adapt(NDTensors.unwrap_array_type(A), to_itensor(v, s))
    if side == :left
        return check_boundary(A * replaceind(vt, s => s'))
    elseif side == :right
        return check_boundary(noprime(A * vt))
    else
        error("Unknown side: $(side) (must be :left or :right)")
    end
end

"""
    fold_boundary(A::ITensor; folded_dim=nothing, bond_ind=nothing)

Fold a boundary state, `A ⊗ conj(A)`, combining physical and bond legs pairwise, and
return it in the boundary convention (so bond dimension χ becomes χ²).

Folding order follows the one used for the `W` tensors (`combiner(ket, bra)`, ket index
fastest), ie. for a vector it gives `kron(conj(v), v)` - **not** `kron(v, conj(v))`. The two
only differ for complex states.

If `folded_dim` is given and already matches the physical dimension of `A`, the state is
assumed to be folded/vectorized already and is returned unchanged (up to retagging).
"""
function fold_boundary(A::ITensor; folded_dim::Union{Int,Nothing}=nothing,
                       bond_ind::Union{Index,Nothing}=nothing, tags="Site,rho0")

    check_boundary(A)
    iP = boundary_phys_ind(A)
    bonds = boundary_bond_inds(A)

    # already folded (vectorized) input
    if !isnothing(folded_dim) && dim(iP) == folded_dim
        return replaceind(A, iP => Index(folded_dim, tags))
    end
    if !isnothing(folded_dim) && dim(iP)^2 != folded_dim
        error("Cannot fold boundary state of physical dimension $(dim(iP)) into $(folded_dim)")
    end

    if isempty(bonds)
        ket = A
        bra = dag(prime(A))
        rho = ket * bra
        Cp = combiner(iP, iP')
        rho = rho * Cp
        return replaceind(rho, combinedind(Cp) => Index(dim(iP)^2, tags))
    end

    # work with fresh unprimed bond indices so that priming the conjugate is safe
    fresh = [Index(dim(bb), "bdry_tmp_$(ii)") for (ii, bb) in enumerate(bonds)]
    ket = replaceinds(A, bonds, fresh)
    bra = dag(prime(ket))
    rho = ket * bra

    Cp = combiner(iP, iP')
    rho = rho * Cp

    combs = [combiner(ff, ff') for ff in fresh]
    for cc in combs
        rho = rho * cc
    end

    s = something(bond_ind, Index(dim(first(bonds))^2, "bdry"))
    newbonds = [prime(s, plev(bb)) for bb in bonds]
    rho = replaceinds(rho, combinedind.(combs), newbonds)

    return check_boundary(replaceind(rho, combinedind(Cp) => Index(dim(iP)^2, tags)))
end

fold_boundary(A::AbstractVector; kwargs...) = fold_boundary(to_itensor(A, "Site"); kwargs...)

###############################################################################
#  Attaching boundary states to a temporal chain
###############################################################################

""" Reset ortho limits of an MPS/MPO after we changed its length. """
function _reset_ortho_lims!(psi::AbstractMPS)
    setleftlim!(psi, 0)
    setrightlim!(psi, length(psi) + 1)
    return psi
end

"""
    _qn_boundary_vector(v, hook)

Build a rank-1 boundary ITensor on (the dual of) a QN index. The state must live in a single
QN block, otherwise it has no definite flux and is not representable - e.g. with `SzParity`
conserved, `up_state` is fine but `plus_state` is not.
"""
function _qn_boundary_vector(v::AbstractVector, hook::Index)
    offset = 0
    occupied = Int[]
    for (bi, qd) in enumerate(space(hook))
        d = last(qd)
        any(!iszero, @view v[(offset+1):(offset+d)]) && push!(occupied, bi)
        offset += d
    end
    length(occupied) <= 1 || error("""
        Boundary state spans $(length(occupied)) QN blocks of $(hook) and therefore has no
        definite flux. With quantum numbers conserved, initial/final states must lie in a
        single symmetry sector (e.g. up/down, not plus/minus). Build the chain without QNs
        if you need a superposition.""")
    return ITensor(v, hook)   # attach_* flips the arrow when it contracts it in
end

""" Normalize user input (Vector/Array/ITensor) to an ITensor living on the chain-end index `hook`. """
function _boundary_itensor(x, hook::Index)
    t = if x isa ITensor
        (hasqns(hook) && !hasqns(x) && ndims(x) == 1) ?
            _qn_boundary_vector(itensor_to_vector(x), hook) : x
    else
        hasqns(hook) ? _qn_boundary_vector(complex(collect(x)), hook) :
                       to_itensor(collect(x), Index(dim(hook), "Site"))
    end
    check_boundary(t)
    ip = boundary_phys_ind(t)
    dim(ip) == dim(hook) || error(
        "Boundary state has physical dimension $(dim(ip)) but the temporal chain expects $(dim(hook))")
    return t
end

function _check_boundary_rank(t::ITensor, psi::AbstractMPS)
    if psi isa MPS && ndims(t) == 3
        error("""A rank-3 (bulk) boundary state adds an extra site with two site legs, which only
                 makes sense for a tMPO. Close one of its bonds with `close_boundary(A, v; side)`
                 to get the rank-2 edge tensor needed by a tMPS.""")
    elseif psi isa MPO && ndims(t) == 2
        error("""A rank-2 (edge) boundary state adds an extra site with a single site leg, which only
                 makes sense for an edge tMPS. Pass the rank-3 bulk tensor to build a tMPO.""")
    end
    return t
end

"""
    attach_boundary_bottom!(psi, bl, hook::Index)

Close the bottom (first) end of a temporal chain, whose dangling link is `hook`, with the
boundary state `bl`. Product states are contracted into `psi[1]`, non-product ones are
prepended as an extra site. See [`boundary_tensor`](@ref) for the conventions.
"""
function attach_boundary_bottom!(psi::AbstractMPS, bl, hook::Index)
    blt = _boundary_itensor(bl, hook)
    _check_boundary_rank(blt, psi)
    blt = adapt(NDTensors.unwrap_array_type(psi[1]), blt)
    ip = boundary_phys_ind(blt)

    if is_product_boundary(blt)
        psi[1] = psi[1] * replaceind(blt, ip => dag(stored_ind(psi[1], hook)))
    else
        pushfirst!(psi.data, replaceind(blt, ip => hook))
        _reset_ortho_lims!(psi)
    end
    return psi
end

"""
    attach_boundary_top!(psi, tr, hook::Index; dagger=false)

Close the top (last) end of a temporal chain, whose dangling link is `hook`, with the
boundary state (or operator) `tr`. `tr` is used **as is** unless `dagger=true`, in which case
it is conjugated - as one wants for a bra ⟨ϕ_f| closing the network.
See [`attach_boundary_bottom!`](@ref).
"""
function attach_boundary_top!(psi::AbstractMPS, tr, hook::Index; dagger::Bool=false)
    trt = _boundary_itensor(tr, hook)
    dagger && (trt = dag(trt))
    _check_boundary_rank(trt, psi)
    trt = adapt(NDTensors.unwrap_array_type(psi[end]), trt)
    ip = boundary_phys_ind(trt)

    if is_product_boundary(trt)
        psi[end] = psi[end] * replaceind(trt, ip => dag(stored_ind(psi[end], hook)))
    else
        push!(psi.data, replaceind(trt, ip => hook))
        _reset_ortho_lims!(psi)
    end
    return psi
end
