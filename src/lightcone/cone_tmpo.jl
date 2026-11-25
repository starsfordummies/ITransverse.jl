""" Builds a folded tMPO extended by `n_ext` sites to the top (ie. end) with the tensor `b.WWl` or `b.WWr`,
depending on whether `LR=:left` or `right`. The input time sites `ts` must be already of the (extended) length.

After rotation 90deg clockwise, should look like (for a :left)
```
     |  |  |  |                     <- p' legs
rho0-o--o--o--o--o--o-fold_op
     |  |  |  |  |  |               <- p legs 
````

and the other way round for :right. So a :left tMPO_ext should have `n_ext` more `p` legs than `p'`

**Bottom extension** (`LR_bottom`, `n_ext_bottom`): the same for the BOTTOM end, i.e. the first
`n_ext_bottom` sites get the edge tensor of side `LR_bottom`. This is the column of a
*double* light cone: at infinite temperature the gates outside the FORWARD cone of a bottom
operator cancel by unitality (U𝟙U† = 𝟙), exactly as those outside the backward cone of the
top operator cancel against the trace, so a column can start late with spatial-boundary
tensors at its foot. `ts` are then the column's own time sites (it need not start at time 1)
and `rho0` closes it at its first site. Needs `nbeta == 0`. `LR` may be omitted when
`n_ext == 0`. With the defaults (`n_ext_bottom = 0`) nothing changes.

`inner_bottom`/`inner_top` pass `inner=true` to the boundary attachment, for a column that
ends a boundary MPS stopping inside the network (see `close_boundary`).
"""
function folded_tMPO_ext(b::FoldtMPOBlocks, ts::Vector{<:Index}; 
    LR::Union{Symbol,Nothing}=nothing, n_ext::Int=1, fold_op = nothing, init_beta_only::Bool=true,
    LR_bottom::Union{Symbol,Nothing}=nothing, n_ext_bottom::Int=0, rho0=b.rho0,
    inner_bottom::Bool=false, inner_top::Bool=false)

    Nt = length(ts)
    Nb = b.tp.nbeta
    
    @assert init_beta_only # extending on imag time not implemented yet, so only accept beta at the bottom 
    @assert Nb + n_ext < Nt # extending on imag time not implemented yet
    n_ext_bottom > 0 && Nb > 0 && error("bottom extension needs nbeta == 0 (got $(Nb))")
    n_ext + n_ext_bottom <= Nt || error("n_ext + n_ext_bottom = $(n_ext + n_ext_bottom) > $(Nt) sites")

    (; WWc, WWc_im, WWl, WWr, iL, iR, iP, iPs) = b 
    _check_qn_time_sites(b, ts)
    
    @assert inds(WWc) == inds(WWc_im)

    _edge(side) = if side == :left
        WWl, 0
    elseif side == :right 
        WWr, 1
    else
        error("Invalid LR =  ($(side))  use :left or :right " )
    end
    WWedge, edge_plev = n_ext > 0 ? _edge(LR) : (nothing, 0)
    WWbot, bot_plev = n_ext_bottom > 0 ? _edge(LR_bottom) : (nothing, 0)

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    WWinds =  (b.iP, b.iPs, b.iL, b.iR)

    dim_virtual_inds = dim(b.iR)

    # Time links
    tl = [sim(iR; tags = "Link,time_fold,l=$(ii-1)") for ii in 1:Nt+1]



    #oo = MPO(fill(WWc, length(ts)))
    oo = MPO(Nt)

    for ii = 1:Nb
        newinds = (ts[ii],   ts[ii]',   tl[ii],   tl[ii+1])
        oo[ii] = _replaceinds_arrow(WWc_im, WWinds, newinds)
    end
    for ii = 1:n_ext_bottom # bottom edge: same single-leg tensors as the top edge
        newinds = (prime(ts[ii], bot_plev),   prime(ts[ii], bot_plev),   tl[ii],   tl[ii+1])
        oo[ii] = _replaceinds_arrow(WWbot, WWinds, newinds)
    end
    for ii = max(Nb, n_ext_bottom)+1:Nt-n_ext
        newinds = (ts[ii],   ts[ii]',   tl[ii],   tl[ii+1])
        oo[ii] = _replaceinds_arrow(WWc, WWinds, newinds)
    end
    for ii = Nt-n_ext+1:Nt # no prime here
        newinds = (prime(ts[ii], edge_plev),   prime(ts[ii], edge_plev),   tl[ii],   tl[ii+1]) 
        oo[ii] = _replaceinds_arrow(WWedge, WWinds, newinds)
    end

   #= 
    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end
    
    for ii = 1:n_ext
        oo[end-ii+1] = WWedge
    end

    for ii in eachindex(oo)
        newinds = (ts[ii],           ts[ii]',          tl[ii],    tl[ii+1])
        oo[ii] = _replaceinds_arrow(oo[ii], WWinds, newinds)
    end

    =# 

    # Contract first tensor with initial state, last one with the operator (default Identity)
    attach_boundary_bottom!(oo, _qn_fold_boundary(b, rho0, tl[1]), tl[1]; inner=inner_bottom)
    attach_boundary_top!(oo, _qn_fold_boundary(b, fold_op, tl[end]), tl[end]; inner=inner_top)

    return oo

end
