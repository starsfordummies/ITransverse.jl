

""" Build folded tMPO with an extra site at the beginning for the initial state.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function folded_tMPO_in(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)

    @assert b.tp.nbeta <= length(ts)
    (; WWc, WWc_im) = b 
    #WWc = b.WWc
    #WWc_im = b.WWc

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


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor `b.WWl`. 
After rotation 90deg clockwise, should look like
```
     |  |  |  |
rho0-o--o--o--o--o-fold_op
     |  |  |  |  |
````

"""
function folded_tMPO_L(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    @assert b.tp.nbeta < length(ts)
    (; WWc, WWc_im) = b 
    #WWc = b.WWc
    #WWc_im = b.WWc

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



""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor `b.WWr`` 
After rotation 90deg clockwise, should look like
```
     |  |  |  |  |
rho0-o--o--o--o--o-fold_op
     |  |  |  |  
````
"""
function folded_tMPO_R(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    @assert b.tp.nbeta < length(ts)

    (; WWc, WWc_im) = b 
    #WWc = b.WWc
    #WWc_im = b.WWc

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


# """ Given an MPO of length N and an MPS (or MPO) of length N-1, extends the target object to the *right* """
# function apply_extend!(w::MPO, psi::Union{MPS,MPO}; cutoff=nothing, maxdim=nothing)
#     @assert length(w) == length(psi) + 1
#     @assert length(w) > 1 

#     last_tensor = pop!(w.data)
#     opsi = apply(w, psi, alg="naive") #, cutoff, maxdim)

#     @info "before: $(length(opsi))"
#     push!(opsi.data, last_tensor)
#     @info "after: $(length(opsi))"

#     # put it back
#     push!(w.data, last_tensor)

#     return opsi
# end


""" tMPO with n_edge boundary tensors. The new time sites `ts` must be already of the (extended) length """
function folded_tMPO_ext(b::FoldtMPOBlocks, ts::Vector{<:Index}; LR::String, n_ext::Int=1, fold_op = nothing)
    @assert b.tp.nbeta + n_ext < length(ts) # extending on imag time not implemented yet
    (; WWc, WWc_im, WWl, WWr) = b 

    WWedge = LR == "L" ? WWl : WWr

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end
    
    for ii = 1:n_ext
        oo[end-ii+1] = WWedge
    end

    # Physical(space) => Virtual(time) index
    virtual_ind = b.rot_inds[:R]

    # Time links
    tl = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    for ii in eachindex(oo)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:L],b.rot_inds[:R] )
        newinds = (ts[ii],           ts[ii]',          tl[ii],    tl[ii+1])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)
    end

    # Contract first tensor with initial state
    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), tl[1])

    # Contract last tensor with operator
    fold_op = something(fold_op, vectorized_identity(dim(virtual_ind)))
    oo[end] *= adapt(dttype, ITensor(fold_op, tl[end]))

    return oo

end
