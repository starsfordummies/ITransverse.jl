

""" Builds a folded tMPO extended by `n_ext` sites to the top (ie. end) with the tensor `b.WWl` or `b.WWr`,
depending on whether `LR=:left` or `right`. The input time sites `ts` must be already of the (extended) length.

After rotation 90deg clockwise, should look like (for a :left)
```
     |  |  |  |                     <- p' legs
rho0-o--o--o--o--o--o-fold_op
     |  |  |  |  |  |               <- p legs 
````

and the other way round for :right. So a :left tMPO_ext should have `n_ext` more `p` legs than `p'`
"""
function folded_tMPO_ext(b::FoldtMPOBlocks, ts::Vector{<:Index}; 
    LR::Symbol, n_ext::Int=1, fold_op = nothing, init_beta_only::Bool=true)

    Nt = length(ts)
    Nb = b.tp.nbeta
    
    @assert init_beta_only # extending on imag time not implemented yet, so only accept beta at the bottom 
    @assert Nb + n_ext < Nt # extending on imag time not implemented yet

    (; WWc, WWc_im, WWl, WWr, rot_inds) = b 
    
    @assert inds(WWc) == inds(WWc_im)


    WWedge, edge_plev = if LR == :left
        WWl, 0
    elseif LR == :right 
        WWr, 1
    else
        error("Invalid LR =  ($(LR))  use :left or :right " )
    end

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:L],b.rot_inds[:R] )

    dim_virtual_inds = dim(b.rot_inds[:R])

    # Time links
    tl = [Index(dim_virtual_inds,"Link,time_fold,l=$(ii-1)") for ii in 1:Nt+1]



    #oo = MPO(fill(WWc, length(ts)))
    oo = MPO(Nt)

    for ii = 1:Nb
        newinds = (ts[ii],   ts[ii]',   tl[ii],   tl[ii+1])
        oo[ii] = replaceinds(WWc_im, WWinds, newinds)
    end
    for ii = Nb+1:Nt-n_ext
        newinds = (ts[ii],   ts[ii]',   tl[ii],   tl[ii+1])
        oo[ii] = replaceinds(WWc, WWinds, newinds)
    end
    for ii = Nt-n_ext+1:Nt # no prime here
        newinds = (prime(ts[ii], edge_plev),   prime(ts[ii], edge_plev),   tl[ii],   tl[ii+1]) 
        oo[ii] = replaceinds(WWedge, WWinds, newinds)
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
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)
    end

    =# 

    # Contract first tensor with initial state
    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), tl[1])

    # Contract last tensor with operator, default to Identity
    fold_op = something(fold_op, vectorized_identity(dim_virtual_inds))
    oo[end] *= adapt(dttype, ITensor(fold_op, tl[end]))

    return oo

end
