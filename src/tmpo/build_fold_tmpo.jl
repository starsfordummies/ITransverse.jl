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
    folded_open_tMPO(WWc, time_sites)
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

    tMPO[1] = ITensor(eltype(WWc)[1,0,0,1] , t1, rot_links[1], t1')
    tMPO[end] = ITensor(eltype(WWc)[1,0,0,1] , tend, rot_links[end], tend')

    return tMPO 

end



""" Given an initial state and a fold_operator as (folded) vectors, build the folded tMPO from those"""
function folded_tMPO(b::FoldtMPOBlocks, fold_op::ITensor, ts::Vector{<:Index})
    
    tMPO = folded_open_tMPO(b.WWc, ts)

    @assert length(fold_op) == linkdims(tMPO)[end]

    init_state_tensor = ITensor(b.rho0.tensor.storage, inds(tMPO[1]))
    fold_op_tensor = ITensor(fold_op, inds(tMPO[end]))

    tMPO.data[1] = init_state_tensor
    tMPO.data[end] = fold_op_tensor

    contract_edges!(tMPO)

    return tMPO

end


# function folded_tMPO(tp::tmpo_params, time_sites::Vector{<:Index};
#      init_state = tp.bl, fold_op = tp.tr)

#     tMPO_blocks = FoldtMPOBlocks(tp, init_state)
#     folded_tMPO(tMPO_blocks, fold_op, time_sites)

# end



function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    s, sp, r, l = inds(b.WWc)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
        # oo[ii] *= delta(s,ts[ii])
        # oo[ii] *= delta(sp,ts[ii]')
        # oo[ii] *= delta(r,ll[ii+1])
        # oo[ii] *= delta(l,ll[ii])
    end

    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= ITensor(fold_op, ll[end])

    return oo

end

function folded_tMPO_L(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    oo[end] = b.WWl
    #s, sp, r, l = inds(b.WWc)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
        # oo[ii] *= delta(s,ts[ii])
        # oo[ii] *= delta(sp,ts[ii]')
        # oo[ii] *= delta(r,ll[ii+1])
        # oo[ii] *= delta(l,ll[ii])
    end

    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= ITensor(fold_op, ll[end])

    return oo

end


function folded_tMPO_R(b::FoldtMPOBlocks, ts::Vector{<:Index}, fold_op::Vector{<:Number} = [1,0,0,1])
    oo = MPO(fill(b.WWc, length(ts)))
    oo[end] = b.WWr
    #s, sp, r, l = inds(b.WWc)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(oo)
        newinds = (ts[ii],ts[ii]',ll[ii+1],ll[ii])
        oo[ii] = replaceinds(oo[ii], inds(b.WWc), newinds)
        # oo[ii] *= delta(s,ts[ii])
        # oo[ii] *= delta(sp,ts[ii]')
        # oo[ii] *= delta(r,ll[ii+1])
        # oo[ii] *= delta(l,ll[ii])
    end

    oo[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    oo[end] *= ITensor(fold_op, ll[end])

    return oo

end



function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWl, length(ts)))
    s, r, l = inds(b.WWl)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], inds(b.WWl), newinds)
        # psi[ii] *= delta(s,ts[ii])
        # psi[ii] *= delta(r,ll[ii+1])
        # psi[ii] *= delta(l,ll[ii])
    end

    psi[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] *= ITensor([1,0,0,1], ll[end])

    return psi 
end


function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index})
    psi = MPS(fill(b.WWr, length(ts)))
    s, r, l = inds(b.WWr)
    ll = [Index(dim(r),"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]
    for ii in eachindex(psi)
        newinds = (ts[ii],ll[ii+1],ll[ii])
        psi[ii] = replaceinds(psi[ii], inds(b.WWr), newinds)
        # psi[ii] *= delta(s,ts[ii])
        # psi[ii] *= delta(r,ll[ii+1])
        # psi[ii] *= delta(l,ll[ii])
    end

    psi[1] *= b.rho0 * delta(ind(b.rho0,1), ll[1])
    psi[end] *= ITensor([1,0,0,1], ll[end])

    return psi 
end
