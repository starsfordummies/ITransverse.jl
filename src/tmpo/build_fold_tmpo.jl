""" Contracts the two edges of an MPO, making a new MPO with length (N-2) """
function contract_edges!(TT::MPO)
    llen = length(TT)

    new_first = TT[1] * TT[2]
    combi = combiner(siteind(TT,1), siteind(TT,2), tags=tags(siteind(TT,2)))
    new_first *= combi
    #new_idx = Index(dim(combinedind(combi), tags(siteind(TT,1))))
    new_first *= combi'

    new_last = TT[end-1] * TT[end]
    combi = combiner(siteind(TT,llen-1), siteind(TT,llen), tags=tags(siteind(TT,llen-1)))
    new_last *= combi
    new_last *= combi'

    pop!(TT.data)
    popfirst!(TT.data)

    TT.data[1] = new_first
    TT.data[end] = new_last

end



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




""" Given `tp` parameters and `time_sites` with Nt sites,
 builds a folded tMPO with rotated indices 
 and additional identity tensors (meant to be replaced) with trivial site indices on the first and last site.
 Final length is Nt + 2 
"""

function folded_open_tMPO(tp::tmpo_params, time_sites::Vector{<:Index})

    WWc = build_WWc(tp)
    folded_open_tMPO(WWc[1], time_sites)
end

""" Given building blocks and time_sites, returns an MPO with two extra sites (one on each side), 
allowing to insert arbitrary operators at a later stage"""
function folded_open_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index})

    oo = MPO(fill(b.WWc, length(ts)))
    ri = ind(b.WWc,3)
    ll = [Index(dim(ri),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    pushfirst!(oo.data, adapt(dttype, ITensor([1,0,0,1], ll[1])))
    push!(oo.data,  adapt(dttype, ITensor([1,0,0,1], ll[end])))

    return oo
end


function folded_open_tMPO(WWc::ITensor, time_sites::Vector{<:Index})
    
    Nsteps = length(time_sites)

    iCwL, iCwR, iCp, iCps = inds(WWc)

    # define the links of the rotated MPO 
    rot_links = [Index( dim(iCp) , "Link,rotl=$(ii-1)") for ii in 1:(Nsteps + 1)]

    # Start building the tMPO
    tMPO = MPO(Nsteps+2)

    # Rotate indices and fill the MPO
    for ii = 1:Nsteps
        tMPO[ii+1] = WWc
        tMPO[ii+1] *= delta(iCwR, time_sites[ii])
        tMPO[ii+1] *= delta(iCwL, time_sites[ii]') 
        tMPO[ii+1] *= delta(iCps, rot_links[ii]) 
        tMPO[ii+1] *= delta(iCp, rot_links[ii+1]) 
    end

    t1 = Index(1,"trivial")
    tend = sim(t1)

    dttype = NDTensors.unwrap_array_type(b.WWc)
    tMPO[1] = adapt(dttype, ITensor(eltype(WWc)[1,0,0,1] , t1, rot_links[1], t1'))
    tMPO[end] = adapt(dttype, ITensor(eltype(WWc)[1,0,0,1] , tend, rot_links[end], tend'))

    return tMPO 

end





""" Given building blocks and time sites, builds folded tMPO associated with `fold_op`. 
Defaults to closing with identity if no operator is specified"""
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    rind = ind(b.WWc,3)
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


###### New ver with beta 
""" Given building blocks and time sites, builds folded tMPO associated with `fold_op`. 
Defaults to closing with identity if no operator is specified"""
function folded_tMPO(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])

    @assert b.tp.nbeta <= length(ts)
    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end

    virtual_ind = ind(b.WWc,3)
    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end

""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor WWl """
function folded_tMPO_L(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    oo[end] = b.WWl
    rind = ind(b.WWc,3)
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor WWl """
function folded_tMPO_L(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    @assert b.tp.nbeta < length(time_sites)
    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end
    
    oo[end] = b.WWl
    virtual_ind = ind(b.WWc,3)
    ll = [Index(dim(virtual_ind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end



function folded_tMPO_R(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    oo[end] = b.WWr
    rind = ind(b.WWc,3)
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor WWl """
function folded_tMPO_R(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    @assert b.tp.nbeta < length(time_sites)

    WWc = b.WWc
    WWc_im = b_im.WWc

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    oo = MPO(fill(WWc, length(ts)))

    for ib = 1:b.tp.nbeta
        oo[ib] = WWc_im
    end

    oo[end] = b.WWr

    rind = ind(b.WWc,3)
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWl, length(ts)))
    s, r, l = inds(b.WWl)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], inds(b.WWl), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    psi[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] *= adapt(dttype, ITensor([1,0,0,1], ll[end]))

    return psi 
end


function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWr, length(ts)))
    s, r, l = inds(b.WWr)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], inds(b.WWr), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    psi[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] *= adapt(dttype, ITensor([1,0,0,1], ll[end]))

    return psi 
end
