""" Builds temporal MPO and starting tMPS guess for *forward evolution only*
with `nbeta` steps of imaginary time regularization.
Closes with initial state again, so it's a Loschmidt echo type setup.

Returns (tMPO, tMPS) pair.

The structure built (Loschmidt style) after rotation is
```
(left[bottom]_state)---Wβ--Wβ---Wt-Wt-Wt-...-Wt---Wβ--Wβ---(right[top]_state)
                      (nbeta)                     (nbeta)
```
The boundary states `bl` (bottom/initial) and `tr` (top/final) can be product states
(vectors) or non-product ones, in which case they add an extra site to the chain -
see [`boundary_tensor`](@ref) for the conventions.

The bottom state `bl` is used as the ket |bl⟩; the top one is **conjugated** by default
(`dagger_tr=true`), so the network built is the amplitude ⟨tr|U…U|bl⟩ - as one wants for a
Loschmidt echo ⟨ψ_0|U^N|ψ_0⟩. Pass `dagger_tr=false` to close with `tr` taken as given
instead. Whichever you choose, use the *same* value for the tMPO and for the edge tMPS, or
the columns will not describe the same network. It only matters for complex `tr`.
 """

function fw_tMPO(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fw_tMPO(b, time_sites; kwargs...)
end



""" Forward tMPO with open top (=right, after rotation) leg, so we can plug anything afterwards.
The dangling top link is tagged `"Link,tr"`. """
function fw_tMPO_opentr(b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl = b.tp.bl,
    nbeta=b.tp.nbeta, init_beta_only::Bool=false)

    tMPO, bl_ind, tr_ind = fw_tMPO_open_edges(b, time_sites; nbeta, init_beta_only)

    tMPO[end] = replaceind(tMPO[end], tr_ind => settags(tr_ind, "Link,tr"))

    # Contract boundary state (bottom/left)
    attach_boundary_bottom!(tMPO, bl, bl_ind)

    return tMPO
end


""" Close an open-top tMPO (see [`fw_tMPO_opentr`](@ref)) with the top boundary state `tr`,
conjugated unless `dagger_tr=false` (see [`fw_tMPO`](@ref)). """
function fw_tMPO(ww::MPO, tr; dagger_tr::Bool=true)
    tr_link = only(inds(ww[end], "Link,tr"))
    attach_boundary_top!(ww, tr, tr_link; dagger=dagger_tr)
    return ww
end




""" Builds forward tMPO with nbeta steps on one side only:
in-U(β)-U(β)-..U(β)-U(idt)-U(idt)-U(idt)-U(idt)-fin
   |___nbeta_____|
   Returns tMPO
"""
function fw_tMPO_initbetaonly(b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl = b.tp.bl, tr, dagger_tr::Bool=true)
    ww = fw_tMPO_opentr(b, time_sites; init_beta_only=true, bl)
    fw_tMPO(ww, tr; dagger_tr)
end



function fw_left_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; kwargs...)
    fw_tMPS(b,time_sites; LR=:left, kwargs...)
end
function fw_right_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; kwargs...)
    fw_tMPS(b,time_sites; LR=:right, kwargs...)
end


function fw_tMPS(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fw_tMPS(b, time_sites; kwargs...)
end

""" Forward (edge) tMPS. `bl`/`tr` are the bottom/top boundary states: product states
(vectors) or rank-2 edge tensors of a non-product boundary MPS, which add one site
to the chain (see [`boundary_tensor`](@ref), [`close_boundary`](@ref)).
`tr` is conjugated unless `dagger_tr=false`; it must match the tMPO it is used with. """
function fw_tMPS(
    b::FwtMPOBlocks,
    time_sites::Vector{<:Index};
    bl = b.tp.bl,
    nbeta = b.tp.nbeta,
    tr,
    dagger_tr::Bool=true,
    LR::Symbol = :right,
    init_beta_only::Bool=false,
    sided::Bool=false
)

    Ntot = length(time_sites)

    @assert nbeta <= Ntot

    # Choose direction-dependent fields and indices
    (W, W_im, iL, iR, iP) = if LR == :left
        (b.Wl, b.Wl_im, b.iL, b.iR, b.iPs)
    elseif LR == :right
        (b.Wr, b.Wr_im, b.iL, b.iR, b.iP)
    else
        error("Unknown LR: $(LR)")
    end

    # Make same indices for real and imag, it's easier afterwards
    replaceinds!(W_im, inds(W_im), inds(W))

    b1,b2 = beta_lims(Ntot, nbeta, init_beta_only)

    rot_links_mps = [sim(iR, tags="Link,rotl=$(ii-1)") for ii in 1:(Ntot + 1)]

    # the right edge column carries the dual of the time site index (see `boundary_tensor`
    # for the analogous statement about the boundary bonds); `dag` is inert without QNs
    site_of(ii) = LR == :right ? dag(time_sites[ii]) : time_sites[ii]

    tMPS = MPS(Ntot)

    for ii = 1:Ntot
        # closing beta block = conjugate of the opening one; `conj`, not `dag`, so the arrows
        # are not reversed with QNs (see the same point in `fw_tMPO_open_edges`)
        Wii = ii <= b1 ? W_im : (ii <= b2 ? W : conj(W_im))
        # take the legs *as stored* (the edge tensors Wl/Wr carry their own arrows) and
        # match the arrow of each replacement; all of this is inert without QNs
        sT, lT, rT = stored_ind(Wii, iP), stored_ind(Wii, iL), stored_ind(Wii, iR)
        tMPS[ii] = Wii * delta(dag(sT), arrow_match(sT, site_of(ii))) *
                   delta(dag(lT), arrow_match(lT, rot_links_mps[ii])) *
                   delta(dag(rT), arrow_match(rT, dag(rot_links_mps[ii+1])))
    end

    # Contract edges with boundary states. A non-product boundary is *appended* as its own
    # site, so count what each end added: afterwards nothing in the MPS distinguishes a
    # boundary site from a time site (see `SidedMPS`).
    nb = length(tMPS)
    attach_boundary_bottom!(tMPS, bl, rot_links_mps[1])
    nbot = length(tMPS) - nb

    nb = length(tMPS)
    attach_boundary_top!(tMPS, tr, rot_links_mps[end]; dagger=dagger_tr)
    ntop = length(tMPS) - nb

    # `sided=true` keeps track of which edge this vector is, see `SidedMPS`
    return sided ? SidedMPS(tMPS, LR, nbot, ntop) : tMPS
end






""" Forward tMPO with open edges, so we can plug anything afterwards.
Returns `(tMPO, bottom_link, top_link)` """
function fw_tMPO_open_edges(b::FwtMPOBlocks, time_sites::Vector{<:Index}; nbeta=b.tp.nbeta, init_beta_only::Bool)

    Ntot = length(time_sites)

    (; Wc, Wc_im, iL, iR, iP, iPs) = b

    @assert nbeta <= Ntot

    b1,b2 = beta_lims(Ntot, nbeta, init_beta_only)

    # Make same indices for real and imag, it's easier aftwards
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    # `sim` (not Index(dim(...))) so QN blocks and arrows survive; `dag` fixes the arrow of
    # the legs that point the other way. Both are no-ops without QNs.
    time_links = [sim(iR, tags="Link,rotl=$(ii-1)") for ii in 1:(Ntot+1)]

    newinds(ii) = (time_sites[ii], dag(time_sites[ii])', dag(time_links[ii]), time_links[ii+1])

    tMPO =  MPO(fill(Wc, Ntot))

    for ii = 1:b1
        tMPO[ii] = replaceinds(Wc_im, (iP, iPs, iL, iR), newinds(ii))
    end
    for ii = b1+1:b2
        tMPO[ii] = replaceinds(Wc, (iP, iPs, iL, iR), newinds(ii))
    end
    # The closing beta block is the *conjugate* of the opening one. Spell that `conj`, not
    # `dag`: with QNs `dag` also reverses every arrow, which flips the two link legs of these
    # tensors and breaks the chain (site b2 would hand an `Out` leg to another `Out`), and
    # then `attach_boundary_top!` cannot contract the hook. `conj` == `dag` without QNs.
    for ii = b2+1:Ntot
        tMPO[ii] = replaceinds(conj(Wc_im), (iP, iPs, iL, iR), newinds(ii))
    end

    return tMPO, time_links[1], time_links[end]

end


function fw_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl = b.tp.bl, tr = b.tp.bl,
    dagger_tr::Bool=true, nbeta=b.tp.nbeta, init_beta_only::Bool=false)

    oo, bl_ind, tr_ind = fw_tMPO_open_edges(b, time_sites; nbeta, init_beta_only)

    attach_boundary_bottom!(oo, bl, bl_ind)
    attach_boundary_top!(oo, tr, tr_ind; dagger=dagger_tr)

    return oo

end
