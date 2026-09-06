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


function folded_left_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPS(b,ts; LR=:left, kwargs...)
end
function folded_right_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; kwargs...)
    folded_tMPS(b,ts; LR=:right, kwargs...)
end

""" Folded (edge) tMPS. `rho0`/`fold_op` are the bottom/top boundary states: (folded)
product states or rank-2 edge tensors of a non-product boundary MPS, which add one site
to the chain (see [`boundary_tensor`](@ref), [`close_boundary`](@ref), [`fold_boundary`](@ref)).

`sided=true` returns a [`SidedMPS`](@ref), which remembers both the side and how many
boundary sites each end added - the only reliable way to tell those apart from time sites
afterwards. """
function folded_tMPS(b::FoldtMPOBlocks, ts::Vector{<:Index}; LR::Symbol=:right,
    init_beta_only::Bool=true, rho0=b.rho0, fold_op=nothing, sided::Bool=false)

    if !init_beta_only
        error("init_beta on both sides not implemented yet")
    end
    if LR == :left
        WW = b.WWl
        WW_im = b.WWl_im
        WWinds = (b.iPs, b.iL, b.iR)
        get_newinds = (ii, tlinks) -> (ts[ii], tlinks[ii], tlinks[ii+1])
    elseif LR == :right
        WW = b.WWr
        WW_im = b.WWr_im
        WWinds = (b.iP, b.iR, b.iL)
        get_newinds = (ii, tlinks) -> (ts[ii], tlinks[ii+1], tlinks[ii])
    else
        error("Unknown LR: $(LR) (must be :left or :right)")
    end

    psi = MPS(fill(WW, length(ts)))
    replaceinds!(WW_im, inds(WW_im), inds(WW))

    for ib = 1:b.tp.nbeta
        psi[ib] = WW_im
    end

    tlinks = [Index(dim(b.iR), "Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    for ii in eachindex(psi)
        newinds = get_newinds(ii, tlinks)
        psi[ii] = replaceinds(psi[ii], WWinds, newinds)
    end

    # A non-product boundary is *appended* as its own site; count what each end added, see
    # the same point in `fw_tMPS`.
    nb = length(psi)
    attach_boundary_bottom!(psi, rho0, tlinks[1])
    nbot = length(psi) - nb

    nb = length(psi)
    attach_boundary_top!(psi, something(fold_op, vectorized_identity(tlinks[end])), tlinks[end])
    ntop = length(psi) - nb

    # `sided=true` keeps track of which edge this vector is, see `SidedMPS`
    return sided ? SidedMPS(psi, LR, nbot, ntop) : psi
end





""" This works for murg construction, need to check how it does with the others... """

""" The *primed* (left-facing) site index of an MPO, ie. the leg we close to turn it into an MPS.
Not simply `siteind(T,j)'`, which assumes the unprimed leg comes first. """
_closing_siteind(T::MPO, j::Int) = only(filter(i -> plev(i) == 1, siteinds(T, j)))

"""Quick way to get init mps from an MPO Murg, just close the corresponding MPO with [1,0,0,0] to one side.
Nornalization might be not the best """
function folded_right_tMPS_murg(T::MPO)

    psi = MPS(deepcopy(T.data))

    dttype = NDTensors.unwrap_array_type(T[1])

    for ii in eachindex(psi)
        psi[ii] *= adapt(dttype, ITensor([1,0,0,0], _closing_siteind(T,ii)))
    end

    return psi
end



"""Quick way to get init mps, just close the corresponding MPO with [1,0,0,0] to one side.
The first site can be a (non-product) boundary state of any dimension, there we close with `e_1`.
Nornalization might be not the best """
function folded_right_tMPS_in_murg(T::MPO)

    psi = MPS(deepcopy(T.data))

    s1 = _closing_siteind(T,1)
    one_first = zeros(dim(s1))
    one_first[1] = 1
    dttype = NDTensors.unwrap_array_type(T[1])
    psi[1] = psi[1] * adapt(dttype, ITensor(one_first, s1))
    for ii in eachindex(psi)[2:end]
        psi[ii] *= adapt(dttype, ITensor([1,0,0,0], _closing_siteind(T,ii)))
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

    (; tp, WWc, WWc_im, iL, iR, iP, iPs) = b

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

    virtual_ind_size = dim(iR)

    # two tlinks will be contracted at the end
    tlinks = [Index(virtual_ind_size,"Link,time_fold,l=$(ii-1)") for ii in 1:length(ts)+1]

    WWinds =  (iP, iPs, iL, iR)

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


""" Builds folded tMPO. Of the `ts` timesites, the first `b.tp.nbeta` ones are imaginary time ones.
 Accepted kwargs: fold_op (default=Identity, accepts Array or ITensor), verbose(=false), init_beta_only(=true).
 `rho0` and `fold_op` may be non-product boundary states, in which case they add one site each
 to the tMPO (see [`boundary_tensor`](@ref)). """
function folded_tMPO(b::FoldtMPOBlocks, ts::Vector{<:Index};
                     fold_op=nothing,
                     init_beta_only::Bool=true,
                     verbose::Bool=false,
                     rho0=b.rho0)

    oo, bl_ind, tr_ind = folded_tMPO_open_edges(b, ts; init_beta_only, verbose)

    attach_boundary_bottom!(oo, rho0, bl_ind)
    attach_boundary_top!(oo, something(fold_op, vectorized_identity(tr_ind)), tr_ind)

    return oo
end