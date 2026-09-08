########################
########## MPO #########
########################


function fwback_tMPO(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fwback_tMPO(b, time_sites; kwargs...)
end

function fwback_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index}; init_beta_only::Bool=false, kwargs...)
    nbeta = b.tp.nbeta

    Ntot = length(time_sites) 
    Nt = Ntot - nbeta

    @assert Nt >= 0 && iseven(Nt)

    Nfw = div(Nt,2)

    betai, betaf = init_beta_only ? (nbeta, 0) : (div(nbeta,2), div(nbeta,2)) 

    fwback_tMPO(b, time_sites, betai, Nfw, Nfw, betaf; kwargs...)
end



function fwback_tMPO_open_edges(b::FwtMPOBlocks, time_sites::Vector{<:Index}, nbetai::Int, nfw::Int, nback::Int, nbetaf::Int; 
    mid_op = [1,0,0,1], t_op::Int=nbetai+nfw)

    @info "Building fwback with $(nbetai)-$(nfw)-$(nback)-$(nbetaf) - operator at $(nbetai+nfw)"

    Ntot = length(time_sites) 
    @assert nbetai + nfw + nback + nbetaf == Ntot

    (; tp, Wc, Wc_im, iL, iR, iP, iPs) = b

    elt = NDTensors.unwrap_array_type(tp.bl)

    ind_op = sim(iR, tags="op")
    ten_mid_op = adapt(elt, ITensor(mid_op, ind_op, ind_op'))

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    # `sim` + `dag` keep QN blocks and arrows; both inert without QNs
    time_links = [sim(iR, tags="Link,rotl=$(ii-1)") for ii in 1:(Ntot+1)]

    tMPO =  MPO(Ntot)

    for ii = 1:nbetai
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(Wc_im, (iP, iPs, iL, iR), (time_sites[ii],dag(time_sites[ii])', dag(time_links[ii]),time_links[ii+1]))
    end
    for ii = nbetai+1:nbetai+nfw
        tMPO[ii] = replaceinds(Wc, (iP, iPs, iL, iR), (time_sites[ii],dag(time_sites[ii])', dag(time_links[ii]),time_links[ii+1]))
    end

    for ii = nbetai+nfw+1:nbetai+nfw+nback
        tMPO[ii] = replaceinds(dag(Wc), (iP, iPs, iL, iR), (dag(time_sites[ii])',time_sites[ii], dag(time_links[ii]),time_links[ii+1]))  # TODO Check [ts',ts] order
    end
    for ii = nbetai+nfw+nback+1:Ntot
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(dag(Wc_im), (iP, iPs, iL, iR), (dag(time_sites[ii])',time_sites[ii], dag(time_links[ii]),time_links[ii+1])) # TODO Check [ts',ts] order
    end


    # Plug operator in the column
    cl = commonind(tMPO[t_op], tMPO[t_op+1])
    tMPO[t_op] = replaceind(contract(tMPO[t_op], ten_mid_op, cl, ind_op), ind_op' => cl)
    #@show inds(tMPO[Nt])
    
    return tMPO, time_links[1], time_links[end]

end



""" Unfolded tMPO with 
- `nbetai` initial steps of imaginary time evolution 
- `nfw` steps of forward time evolution 
-  (optionally) a `mid_op` operator insertion 
- `nback` steps of backwards time evolution
- `nbetaf` steps of imaginary time evolution

The top boundary `tr` closes the backward branch, ie. it is the bra ⟨tr|, so it is
conjugated by default (`dagger_tr=true`); pass `dagger_tr=false` to use it as given.
Use the same value for the tMPO and for the edge tMPS built with `fwback_tMPS`.
"""
function fwback_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index}, nbetai::Int, nfw::Int, nback::Int, nbetaf::Int; 
    bl = b.tp.bl, tr = b.tp.bl, dagger_tr::Bool=true, kwargs...)
    oo, bl_ind, tr_ind = fwback_tMPO_open_edges(b, time_sites, nbetai, nfw, nback, nbetaf; kwargs...)

    attach_boundary_bottom!(oo, bl, bl_ind)
    attach_boundary_top!(oo, tr, tr_ind; dagger=dagger_tr)

    return oo
end


################# 
#### MPS ########
#################


function fwback_tMPS(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fwback_tMPS(b, time_sites; kwargs...)
end

function fwback_tMPS(
    b::FwtMPOBlocks,
    time_sites::Vector{<:Index};
    bl = b.tp.bl,
    tr,
    dagger_tr::Bool=true,
    LR::Symbol = :right,
    init_beta_only::Bool=false
)


    Ntot = length(time_sites)

    tp = b.tp
    nbeta = tp.nbeta

    @assert nbeta <= Ntot

    # Same convention as fwback_tMPO: nbeta imag steps, then Nfw forward and Nfw backward
    Nt = Ntot - nbeta
    @assert Nt >= 0 && iseven(Nt)
    Nfw = div(Nt, 2)
    betai, betaf = init_beta_only ? (nbeta, 0) : (div(nbeta,2), div(nbeta,2))
    @assert betai + 2*Nfw + betaf == Ntot

    # Choose direction-dependent fields and indices
    (W, W_im, iL, iR, iP) = if LR == :left
        (b.Wl, b.Wl_im, b.iL, b.iR, b.iPs)
    elseif LR == :right
        (b.Wr, b.Wr_im, b.iL, b.iR, b.iP)
    else
        error("Unknown LR: $(LR)")
    end

    # Make same indices for real and imag, it's easier afterwards
    replaceinds!(W_im, inds(W_im), inds(W))

    rot_links_mps = [sim(iR, tags="Link,rotl=$(ii-1)") for ii in 1:(Ntot + 1)]
    site_of(ii) = LR == :right ? dag(time_sites[ii]) : time_sites[ii]

    tMPS = MPS(Ntot)

    for ii = 1:Ntot
        Wii = if ii <= betai
            W_im
        elseif ii <= betai + Nfw
            W
        elseif ii <= betai + 2*Nfw
            dag(W)
        else
            dag(W_im)
        end
        sT, lT, rT = stored_ind(Wii, iP), stored_ind(Wii, iL), stored_ind(Wii, iR)
        tMPS[ii] = Wii * delta(dag(sT), arrow_match(sT, site_of(ii))) *
                   delta(dag(lT), arrow_match(lT, rot_links_mps[ii])) *
                   delta(dag(rT), arrow_match(rT, dag(rot_links_mps[ii+1])))
    end

    # Contract edges with boundary states (a non-product one is appended as its own site).
    attach_boundary_bottom!(tMPS, bl, rot_links_mps[1])

    attach_boundary_top!(tMPS, tr, rot_links_mps[end]; dagger=dagger_tr)

    return TransverseMPS(tMPS, LR)
end
