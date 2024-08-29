""" Contracts the edges of the MPO - either the `first` or `last` one or both, if specified
How the indices are combined is still open to changes.. """
function contract_edges!(TT::MPO; first::Bool=false, last::Bool=false)
    llen = length(TT)

    if first
        new_first = TT[1] * TT[2]
        combi = combiner(siteind(TT,1), siteind(TT,2), tags=tags(siteind(TT,2)))
        new_first *= combi
        #new_idx = Index(dim(combinedind(combi), tags(siteind(TT,1))))
        new_first *= combi'

        popfirst!(TT.data)
        TT.data[1] = new_first
    end

    if last
        new_last = TT[end-1] * TT[end]
        # TODO maybe figure out better logic for when combining indices is necessary..
        if ndims(TT[end]) > 2
            combi = combiner(siteind(TT,llen-1), siteind(TT,llen), tags=tags(siteind(TT,llen-1)))
            new_last *= combi
            new_last *= combi'
        end

        pop!(TT.data)
        TT.data[end] = new_last
    end

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
 builds a folded tMPO  *of length  Nt + 2* with rotated indices 
 and additional identity tensors (meant to be replaced) with trivial site indices on the first and last site.

"""
function folded_open_tMPO(tp::tMPOParams, time_sites::Vector{<:Index})

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

    dttype = NDTensors.unwrap_array_type(WWc)
    tMPO[1] = adapt(dttype, ITensor(eltype(WWc)[1,0,0,1] , t1, rot_links[1], t1'))
    tMPO[end] = adapt(dttype, ITensor(eltype(WWc)[1,0,0,1] , tend, rot_links[end], tend'))

    return tMPO 

end





""" Given building blocks and time sites, builds folded tMPO associated with `fold_op`. 
Defaults to closing with identity if no operator is specified"""
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
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
function folded_tMPO(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])

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


""" Build folded tMPO with an extra site at the beginning for the initial state.
The extra site should already be incorporated in the `ts` index list (and we check whether the dimensions match),
 so effectively we're building a tMPO for Nt = length(ts)-1 timesteps  """
function folded_tMPO_in(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])

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



    oo[end] *= adapt(dttype, ITensor(fold_op, ll[end]))

    return oo

end


# Quick way to get init mps, just close with [1,0,0,..] to one side
function folded_right_tMPS_in(T::MPO)

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
function folded_left_tMPS_in(T::MPO)
    return folded_right_tMPS_in(T)
end

""" Builds a folded tMPO extended by one site to the top (ie. end) with the tensor WWl """
function folded_tMPO_L(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
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
function folded_tMPO_L(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
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



function folded_tMPO_R(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector = [1,0,0,1])
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
function folded_tMPO_R(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::AbstractVector  = [1,0,0,1])
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

    rind = ind(b.WWc,3)
    ll = [Index(dim(rind),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(WWc), newinds)
    end

    dttype = NDTensors.unwrap_array_type(b.WWc)
    oo[1] = oo[1] * b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] = oo[end] * adapt(dttype, ITensor(fold_op, ll[end]))

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

    #@show b.WWr
    #@show ts
    dttype = NDTensors.unwrap_array_type(b.WWc)
    psi[1] = psi[1] * b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] = psi[end] * adapt(dttype, ITensor([1,0,0,1], ll[end]))

    return psi 
end

function open_tmpo_top(b::FoldtMPOBlocks, b_im::FoldtMPOBlocks, ts::Vector{<:Index})

    ts = deepcopy(ts)
    push!(ts, sim(ts[end]))
    tmpo = folded_tMPO(b, b_im, ts, [1,0,0,1])
    final_link = linkinds(tmpo)[end]
    pop!(tmpo.data)
    push!(tmpo.data, delta(ts[end],final_link))
    ts[end] = final_link

    return tmpo

end