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




""" Accepted kwargs: fold_op, outputlevel """ 
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)

    outputlevel::Int = get(kwargs,:outputlevel, 0)
    #@info outputlevel
    outputlevel > 1 && @info "Building folded tMPO for (im+real) $(b.tp.nbeta)+$(length(ts)-b.tp.nbeta) sites "

    @assert b.tp.nbeta <= length(ts)

    WWc = b.WWc
    
    WWc_im = b.WWc
    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end

    virtual_ind = b.rot_inds[:R]

    tlinks = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    #@info ll 
    for ii in eachindex(oo)
        WWinds =  (b.rot_inds[:P],b.rot_inds[:Ps],b.rot_inds[:L], b.rot_inds[:R], )
        newinds = (ts[ii],           ts[ii]',          tlinks[ii],      tlinks[ii+1])
        oo[ii] = replaceinds(oo[ii], WWinds, newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] = oo[1] * replaceind(b.rho0, ind(b.rho0,1), tlinks[1])

    # s = inds(b.WWc)[4]
  
    fold_op = get(kwargs, :fold_op, vectorized_identity(dim(virtual_ind)))

    fold_op = to_itensor(fold_op, tlinks[end])
  
    oo[end] = oo[end] * adapt(dttype, fold_op)

    return oo

end


""" Builds a tMPS using the WWl tensors in `b` """ 
function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    (; WWl, WWl_im) = b 

    psi = MPS(fill(WWl, length(ts)))
    
    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWl_im, inds(WWl_im), inds(WWl))

     for ib = 1:b.tp.nbeta
        psi[ib] = WWl_im
    end
    # s, r, l = inds(b.WWl)

    tlinks = [Index(dim(b.rot_inds[:R]),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    for ii in eachindex(psi)

        WWinds =  (b.rot_inds[:P],  b.rot_inds[:L] , b.rot_inds[:R])
        newinds = (ts[ii],           tlinks[ii],      tlinks[ii+1] )   
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    dttype = NDTensors.unwrap_array_type(WWl)
    psi[1] *= replaceind(b.rho0, ind(b.rho0,1), tlinks[1])

    fold_op = get(kwargs, :fold_op, vectorized_identity(tlinks[end]))
    psi[end] = psi[end] * adapt(dttype, to_itensor(fold_op, tlinks[end]))

    return psi 
end

""" Builds a tMPS using the WWr tensors in `b` 
 p' indices are converted to unprimed p for the physical inds """ 
function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)

    (; WWr, WWr_im) = b 
 
    psi = MPS(fill(WWr, length(ts)))
      
    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWr_im, inds(WWr_im), inds(WWr))

     for ib = 1:b.tp.nbeta
        psi[ib] = WWr_im
    end
    
    tlinks = [Index(dim(b.rot_inds[:R]),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    for ii in eachindex(psi)
        WWinds =  (b.rot_inds[:Ps],b.rot_inds[:R],b.rot_inds[:L] )
        newinds = (ts[ii],         tlinks[ii+1],       tlinks[ii])
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    #@show b.WWr
    #@show ts
    dttype = NDTensors.unwrap_array_type(b.WWr)
    psi[1] = psi[1] * b.rho0 * delta(ind(b.rho0,1), tlinks[1])
    

    fold_op = get(kwargs, :fold_op, vectorized_identity(tlinks[end]))
    #@info fold_op

    #@info fold_op
    psi[end] = psi[end] * adapt(dttype, to_itensor(fold_op, tlinks[end]))

    return psi 
end


""" Puts imaginary time on *both* edges of the folded tMPO """
function folded_tMPO_doublebeta(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
    # TODO NEED TO UPDATE TO NEWER CONVENTIONS 
    @assert 2*b.tp.nbeta <= length(ts)
    (; WWc, WWc_im) = b 
    #WWc = b.WWc
    #WWc_im = b.WWc
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
