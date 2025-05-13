""" Build a *rotated and folded* TMPO associated with exp. value starting from eH tensors of U=exp(iHt) 
(defined as a regular spatial MPO on space indices). tMPO is defined on `time_sites`

We rotate our space vectors to the *right* by 90°, ie 

```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```

and contract with  the initial state `init_state` on the *left* and the operator `fold_op` on the *right*

````
             p'
         |   |   |   |
[rho0]==(W)=(W)=(W)=(W)==[operator]
         |   |   |   |
             p
````
"""


#Legacy, remove eventually ?
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
    @assert b.tp.nbeta == 0
    folded_tMPO(b,b, ts, fold_op)
end


###### New ver with beta 
""" Given building blocks and time sites, builds folded tMPO associated with `fold_op`. 
Defaults to closing with identity if no operator is specified. Builds `b.tp.nbeta` steps of b_im blocks
    at the beginning of the tMPO. """
function folded_tMPO(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector)
    folded_tMPO(b, b_im, ts; fold_op=fold_op)
end

function folded_tMPO(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)

    outputlevel::Int = get(kwargs,:outputlevel, 0)
    #@info outputlevel
    outputlevel > 1 && @info "Building folded tMPO for (im+real) $(b.tp.nbeta)+$(length(ts)-b.tp.nbeta) sites "

    @assert b.tp.nbeta <= length(ts)

    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end

    virtual_ind = b.rot_inds[:R]

    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],           ts[ii]',          ll[ii+1],    ll[ii])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])

    # s = inds(b.WWc)[4]
  
    fold_op = get(kwargs, :fold_op, vectorized_identity(dim(virtual_ind)))

    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


""" Build folded tMPO with an extra site at the beginning for the initial state.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function folded_tMPO_in(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)

    @assert b.tp.nbeta <= length(ts)
    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))


    virtual_ind = ind(b.WWc,3)

    # Either rho0 is a 1-dim tensor, 
    # or a 3-legged tensor with legs ordered (left-right-phys)
    #@show dim(virtual_ind) 
    #@show dim(inds(b.rho0))[end]

    @assert dim(virtual_ind) == dim(inds(b.rho0)[end])

    # TODO CHECK HERE WE SHOULD MAKE SURE THAT ROTATED INDICES ARE IN ORDER (L,R,PHYS)
    #@show dims(b.rho0)
    #@show dims(ts)
    if ndims(b.rho0) > 1
        @assert dim(b.rho0,1) ==  dim(b.rho0,2)  # L-R invariance 
        @assert dim(ts[1]) == dim(b.rho0,1)
    end
    # TODO 
    # else
    # b.rho0 *= delta(ts[1],ts[1]')
    # end

    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)]

    oo = MPO(fill(WWc, length(ts)))

    if ndims(b.rho0) == 1
        oo[1] = b.rho0 * delta(inds(b.rho0)[end], ll[1]) 
    elseif ndims(b.rho0) == 3
        oo[1] = b.rho0 * delta(inds(b.rho0)[end], ll[1])
        oo[1] = replaceinds(oo[1], uniqueinds(oo[1], ll[1]), (ts[1],ts[1]') )
    end


    for ib = 2:b.tp.nbeta
        oo[ib] = WWc_im
    end

    for ii in eachindex(oo)[2:end]
        newinds = (ts[ii],ts[ii]',ll[ii],ll[ii-1])
        oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end


    dttype = NDTensors.unwrap_array_type(b.WWc)



    fold_op = get(kwargs, :fold_op, vectorized_identity(dim(virtual_ind)))

    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor WWl """
function folded_tMPO_L(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPO_L(b, b, ts; kwargs...)
end


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor `b.WWl` """
function folded_tMPO_L(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    @assert b.tp.nbeta < length(ts)
    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end
    
    oo[end] = b.WWl
    virtual_ind = b.rot_inds[:R]
    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],           ts[ii]',          ll[ii+1],    ll[ii])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)

        # newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        # oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])


    fold_op = get(kwargs, :fold_op, vectorized_identity(dim(virtual_ind)))
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end



function folded_tMPO_R(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPO_R(b, b, ts; kwargs...)
end


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor `b.WWr`` """
function folded_tMPO_R(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    @assert b.tp.nbeta < length(ts)

    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end

    oo[end] = b.WWr

    rind = b.rot_inds[:R]
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],           ts[ii]',          ll[ii+1],    ll[ii])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)

        # newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        # oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] = oo[1] * b.rho0 * delta(ind(b.rho0,1), ll[1])

    fold_op = get(kwargs, :fold_op, vectorized_identity(dim(rind)))
    oo[end] = oo[end] * adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


""" Builds a tMPS using the WWl tensors in `b` """ 
function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWl, length(ts)))
    # s, r, l = inds(b.WWl)

    ll = [Index(dim(b.rot_inds[:R]),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        # newinds = (ts[ii],ll[ii+1],ll[ii])
        # psi[ii] = replaceinds(psi[ii], inds(b.WWl), newinds)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWl)
    psi[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] *= adapt(dttype, vectorized_identity(ll[end]))

    return psi 
end

""" Builds a tMPS using the WWr tensors in `b` """ 
function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWr, length(ts)))
    #s, r, l = inds(b.WWr)
    ll = [Index(dim(b.rot_inds[:R]),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        # newinds = (ts[ii],ll[ii+1],ll[ii])
        # psi[ii] = replaceinds(psi[ii], inds(b.WWr), newinds)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    #@show b.WWr
    #@show ts
    dttype = NDTensors.unwrap_array_type(b.WWc)
    psi[1] = psi[1] * b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] = psi[end] * adapt(dttype, vectorized_identity(ll[end]))

    return psi 
end


""" Puts imaginary time on *both* edges of the folded tMPO """
function folded_tMPO_doublebeta(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])

    @assert 2*b.tp.nbeta <= length(ts)
    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
        oo[end-ib+1] = WWc_im
    end

    virtual_ind = ind(b.WWc,3)
    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        # newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        # oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],           ts[ii]',          ll[ii+1],    ll[ii])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)

    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end



""" This works for murg construction, need to check how it does with the others... """

"""Quick way to get init mps from an MPO Murg, just close the corresponding MPO with [1,0,0,0] to one side. 
Nornalization might be not the best """
function folded_right_tMPS_murg(T::MPO)

    psi = MPS(deepcopy(T.data))

    dttype = NDTensors.unwrap_array_type(T[1])
 
    for ii in eachindex(psi)
        psi[ii] *= adapt(dttype, ITensor([1,0,0,0], siteind(T,ii)'))
    end

    return psi 
end



"""Quick way to get init mps, just close the corresponding MPO with [1,0,0,0] to one side. 
Nornalization might be not the best """
function folded_right_tMPS_in_murg(T::MPO)

    psi = MPS(deepcopy(T.data))

    one_first = zeros(dim(siteind(T,1)))
    one_first[1] = 1
    dttype = NDTensors.unwrap_array_type(T[1])
    psi[1] = psi[1] * adapt(dttype, ITensor(one_first, siteind(T,1)'))
    for ii in eachindex(psi)[2:end]
        psi[ii] *= adapt(dttype, ITensor([1,0,0,0], siteind(T,ii)'))
    end
    return psi 
end


#TODO 
function folded_left_tMPS_in_murg(T::MPO)
    return folded_right_tMPS_in_murg(T)
end
