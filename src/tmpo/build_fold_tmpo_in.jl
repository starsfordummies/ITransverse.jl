"""
    tMPO_in(b, ts; init_tensor, init_physidx, left=nothing, right=nothing, kwargs...)

Build a tMPO (folded or forward, depending on `b`) on the `length(ts)` time sites `ts`,
with a **non-product initial state**: the tensor `init_tensor` (one column of the initial
boundary MPS, e.g. a tensor of a DMRG ground state) is added as an extra site at the
bottom, so the resulting tMPO has `length(ts)+1` sites.

`init_physidx` is the physical (space) index of `init_tensor`, `left`/`right` its bond
indices - if not given they are taken in the order they appear in the tensor, which is
error prone, so better pass them explicitly. For `FoldtMPOBlocks` the tensor is folded
(`A ⊗ conj(A)`) here.

The boundary bond index of the resulting tMPO is the (unprimed) `"bdry"`-tagged index built by
[`boundary_tensor`](@ref); use [`close_boundary`](@ref) on the same tensor to build matching
edge tMPS.

!!! note
    Each call builds (and, for the folded case, folds) its own boundary tensor, hence its own
    boundary bond index. To get several columns sharing it - e.g. an operator column and an
    identity column - build the boundary state once and pass it directly:
    `bl = boundary_tensor(A; phys, left, right)`, then `fw_tMPO(b, ts; bl, ...)`, or for the
    folded case `rho0 = fold_boundary(bl; folded_dim=dim(b.iL))` and `folded_tMPO(b, ts; rho0, ...)`.
"""
function tMPO_in(b, ts::Vector{<:Index}; init_tensor::ITensor, init_physidx::Index,
                 left=nothing, right=nothing, kwargs...)

    bl = _boundary_from_column(init_tensor, init_physidx, left, right)

    if b isa FoldtMPOBlocks
        rho0 = fold_boundary(bl; folded_dim=dim(b.iL), tags="Site,rho0")
        return folded_tMPO(b, ts; rho0, kwargs...)
    elseif b isa FwtMPOBlocks
        return fw_tMPO(b, ts; bl, kwargs...)
    else
        error("Unknown tMPO blocks type $(typeof(b))")
    end
end

""" Folded tMPO with a non-product initial state, see [`tMPO_in`](@ref) """
folded_tMPO_in(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...) = tMPO_in(b, ts; kwargs...)

""" Forward (unfolded) tMPO with a non-product initial state, see [`tMPO_in`](@ref) """
fw_tMPO_in(b::FwtMPOBlocks, ts::Vector{<:Index}; kwargs...) = tMPO_in(b, ts; kwargs...)


""" Put one column of a boundary MPS in the boundary-state convention, guessing the
left/right bonds from the index order if they are not given explicitly. """
function _boundary_from_column(A::ITensor, phys::Index, left, right)
    if isnothing(left) && isnothing(right)
        bonds = uniqueinds(inds(A), phys)
        # already in the convention (bonds are a prime pair): keep the bond index as is
        if length(bonds) == 2 && noprime(bonds[1]) == noprime(bonds[2])
            return check_boundary(settags(A, "Site", phys))
        end
        if length(bonds) == 2
            left, right = bonds
        elseif length(bonds) == 1
            right = only(bonds)   # single bond: edge column, side is irrelevant
        elseif !isempty(bonds)
            error("Boundary column tensor has $(length(bonds)) bonds besides $(phys)")
        end
    end
    return boundary_tensor(A; phys, left, right)
end
