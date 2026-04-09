""" Build a *rotated and folded* TMPO associated with exp. value starting from eH tensors of U=exp(iHt) 
(inputted as a regular spatial MPO on space indices). 

tMPO is defined on `time_sites`

We rotate our space vectors to the *right* by 90°, ie 

```
   |p'             |L => new p'
L--o--R   =>    p--o--p' => new R
   |p              |R => new p 
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




""" Builds folded tMPO. Of the `ts` timesites, the first `b.tp.nbeta` ones are imaginary time ones.
 Accepted kwargs: fold_op(default=Identity op.), verbose(=false), init_beta_only(=true) """ 
function folded_tMPO_old(b::FoldtMPOBlocks, ts::Vector{<:Index}; fold_op=nothing, init_beta_only::Bool=true, verbose::Bool=false, rho0=b.rho0)

    (; tp, WWc, WWc_im, rot_inds) = b

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    Ntot = length(ts)
    nbeta = tp.nbeta 

    @assert nbeta <= length(ts)

    (b1, b2) = if init_beta_only 
        nbeta, Ntot
    else # beta at the beginning and at the end
        @assert iseven(nbeta)
        beta_half = div(nbeta,2)
        beta_half, Ntot - beta_half 
    end

    if verbose
        @info "Building folded tMPO for (im+real) $(b1)-$(b2)-$(Ntot)) sites "
    end

    oo = MPO(Ntot)

    virtual_ind_size = dim(rot_inds[:R])

    # two tlinks will be contracted at the end
    tlinks = [Index(virtual_ind_size,"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    WWinds =  (rot_inds[:P], rot_inds[:Ps], rot_inds[:L], rot_inds[:R] )

    for ii in eachindex(oo)
        newinds = (ts[ii],        ts[ii]',       tlinks[ii],   tlinks[ii+1])
        if ii > b1 && ii <= b2
            oo[ii] = replaceinds(WWc, WWinds, newinds)
        else
            #@warn "Filling imag beta tensor O[$(ii)]"
            oo[ii] = replaceinds(WWc_im, WWinds, newinds)
        end
    end


    oo[1] = oo[1] * replaceind(rho0, ind(rho0,1), tlinks[1])

    # s = inds(b.WWc)[4]
  
    fold_op = something(fold_op, vectorized_identity(tlinks[end]))

    if verbose
        @info "fold_op = $(vector(fold_op))"
    end

    fold_op = to_itensor(fold_op, tlinks[end])
  
    dttype = NDTensors.unwrap_array_type(WWc)
    oo[end] = oo[end] * adapt(dttype, fold_op)

    return oo

end


function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPS(b,ts; LR=:left, kwargs...)
end
function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPS(b,ts; LR=:right, kwargs...)
end

function folded_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; LR::Symbol=:right, init_beta_only::Bool=true, kwargs...)

    if !init_beta_only
        error("init_beta on both sides not implemented yet")
    end
    if LR == :left
        WW = b.WWl
        WW_im = b.WWl_im
        WWinds = (b.rot_inds[:Ps], b.rot_inds[:L], b.rot_inds[:R])
        get_newinds = (ii, tlinks) -> (ts[ii], tlinks[ii], tlinks[ii+1])
        edge_contract = (psi, b, tlinks) -> psi[1] *= replaceind(b.rho0, ind(b.rho0,1), tlinks[1])
    elseif LR == :right
        WW = b.WWr
        WW_im = b.WWr_im
        WWinds = (b.rot_inds[:P], b.rot_inds[:R], b.rot_inds[:L])
        get_newinds = (ii, tlinks) -> (ts[ii], tlinks[ii+1], tlinks[ii])
        edge_contract = (psi, b, tlinks) -> psi[1] = psi[1] * b.rho0 * delta(ind(b.rho0,1), tlinks[1])
    else
        error("Unknown LR: $(LR) (must be :left or :right)")
    end

    psi = MPS(fill(WW, length(ts)))
    replaceinds!(WW_im, inds(WW_im), inds(WW))

    for ib = 1:b.tp.nbeta
        psi[ib] = WW_im
    end

    tlinks = [Index(dim(b.rot_inds[:R]), "Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    for ii in eachindex(psi)
        newinds = get_newinds(ii, tlinks)
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    dttype = NDTensors.unwrap_array_type(WW)
    edge_contract(psi, b, tlinks)

    fold_op = get(kwargs, :fold_op, vectorized_identity(tlinks[end]))
    psi[end] = psi[end] * adapt(dttype, to_itensor(fold_op, tlinks[end]))

    return psi
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


#TODO non-symm LR 
function folded_left_tMPS_in_murg(T::MPO)
    return folded_right_tMPS_in_murg(T)
end


""" Builds folded tMPO. Of the `ts` timesites, the first `b.tp.nbeta` ones are imaginary time ones.
 Accepted kwargs: fold_op(default=Identity op.), verbose(=false), init_beta_only(=true) """ 
function folded_tMPO_open_edges(b::FoldtMPOBlocks, ts::Vector{<:Index}; init_beta_only::Bool=true, verbose::Bool=false)

    (; tp, WWc, WWc_im, rot_inds) = b

    #match indices for real-imag so it's easier to work with them 
    replaceinds!(WWc_im, inds(WWc_im), inds(WWc))

    Ntot = length(ts)
    nbeta = tp.nbeta 

    @assert nbeta <= length(ts)

    (b1, b2) = if init_beta_only 
        nbeta, Ntot
    else # beta at the beginning and at the end
        @assert iseven(nbeta)
        beta_half = div(nbeta,2)
        beta_half, Ntot - beta_half 
    end

    if verbose
        @info "Building folded tMPO for (im+real) $(b1)-$(b2)-$(Ntot)) sites "
    end

    oo = MPO(Ntot)

    virtual_ind_size = dim(rot_inds[:R])

    # two tlinks will be contracted at the end
    tlinks = [Index(virtual_ind_size,"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    WWinds =  (rot_inds[:P], rot_inds[:Ps], rot_inds[:L], rot_inds[:R] )

    for ii in eachindex(oo)
        newinds = (ts[ii],        ts[ii]',       tlinks[ii],   tlinks[ii+1])
        if ii > b1 && ii <= b2
            oo[ii] = replaceinds(WWc, WWinds, newinds)
        else
            #@warn "Filling imag beta tensor O[$(ii)]"
            oo[ii] = replaceinds(WWc_im, WWinds, newinds)
        end
    end

    return oo, tlinks[1], tlinks[end]

end


#= old 
""" Builds folded tMPO. Of the `ts` timesites, the first `b.tp.nbeta` ones are imaginary time ones.
 Accepted kwargs: fold_op(default=Identity op.), verbose(=false), init_beta_only(=true) """ 
function folded_tMPO_o1(b::FoldtMPOBlocks, ts::Vector{<:Index}; fold_op=nothing, init_beta_only::Bool=true, verbose::Bool=false, rho0=b.rho0)


    oo, bl_ind, tr_ind = folded_tMPO_open_edges(b,ts; init_beta_only, verbose)


    fold_op = something(fold_op, vectorized_identity(Index(dim(tr_ind), "Site")))

    if ndims(rho0) == 1
        oo[1] = contract(oo[1], rho0, bl_ind, only(inds(rho0)))
    else
        @show inds(rho0)
        pushfirst!(oo.data, replaceind(rho0, only(inds(rho0, "Site")) => bl_ind)) 
    end

    if ndims(fold_op) == 1

        dttype = NDTensors.unwrap_array_type(oo[end])
        fold_op = adapt(dttype, fold_op)
        oo[end] = contract(oo[end], fold_op, tr_ind, only(inds(fold_op)))
    else
        push!(oo.data, replaceind(fold_op, only(inds(fold_op, "Site")) => tr_ind))
    end


    if verbose
        @info "fold_op = $(vector(fold_op))"
    end


    return oo

end
=#

"""Convert input to ITensor mapped onto target index, or return identity if nothing."""
function _to_itensor_or_identity(op, target_ind::Index)
    op === nothing && return vectorized_identity(Index(dim(target_ind), "Site"))
    op isa ITensor  && return op
    # Array branch: adapt to correct storage type and wrap
    return ITensor(op, Index(dim(target_ind), "Site"))
end

"""Attach a (possibly vector/matrix) operator to the boundary of an MPS."""
function _attach_boundary_op!(oo::MPO, op::ITensor, bond_ind::Index)
    if ndims(op) == 1
        dttype = NDTensors.unwrap_array_type(oo[end])
        oo[end] = contract(oo[end], adapt(dttype, op), bond_ind, only(inds(op)))
    else
        push!(oo.data, replaceind(op, only(inds(op, "Site")) => bond_ind))
    end
end

"""Attach initial state rho0 to the left boundary of an MPS."""
function _attach_rho0!(oo::MPO, rho0::ITensor, bond_ind::Index)
    if ndims(rho0) == 1
        oo[1] = contract(oo[1], rho0, bond_ind, only(inds(rho0)))
    else
        pushfirst!(oo.data, replaceind(rho0, only(inds(rho0, "Site")) => bond_ind))
    end
end

""" Builds folded tMPO. Of the `ts` timesites, the first `b.tp.nbeta` ones are imaginary time ones.
 Accepted kwargs: fold_op (default=Identity, accepts Array or ITensor), verbose(=false), init_beta_only(=true) """ 
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index};
                     fold_op=nothing,
                     init_beta_only::Bool=true,
                     verbose::Bool=false,
                     rho0=b.rho0)

    oo, bl_ind, tr_ind = folded_tMPO_open_edges(b, ts; init_beta_only, verbose)

    fold_op_tensor = _to_itensor_or_identity(fold_op, tr_ind)

    _attach_rho0!(oo, rho0, bl_ind)
    _attach_boundary_op!(oo, fold_op_tensor, tr_ind)

    verbose && @info "fold_op = $(vector(fold_op_tensor))"

    return oo
end