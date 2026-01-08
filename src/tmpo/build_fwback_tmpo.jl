
function fwback_tMPO(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fwback_tMPO(b, time_sites; kwargs...)
end


""" Forward tMPO with open top (=right, after rotation) leg, so we can plug anything afterwards """
function fwback_tMPO_opentr(b::FwtMPOBlocks, time_sites::Vector{<:Index}; mid_op = [1,0,0,1], bl::ITensor = b.tp.bl, init_beta_only::Bool=false)

    Ntot = length(time_sites) 
    @assert Ntot % 2 == 0 
    Nt = div(length(time_sites),2)

    (; tp, Wc, Wc_im, rot_inds) = b
    nbeta = tp.nbeta 

    @assert nbeta <= Nt

    b1,b2 = beta_lims(Ntot, nbeta, init_beta_only)

    (icL, icR, icP, icPs) = (rot_inds[:L], rot_inds[:R], rot_inds[:P], rot_inds[:Ps]) 

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    time_links = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Ntot-1)]

    tMPO =  MPO(fill(Wc, Ntot))

    for ii = 1:b1
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(Wc_im, (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end
    for ii = b1+1:Nt
        tMPO[ii] = replaceinds(Wc, (icP, icPs), (time_sites[ii],time_sites[ii]') )
    end
    @show inds(tMPO[Nt])
    ten_mid_op = adapt(NDTensors.unwrap_array_type(tMPO[Nt]), ITensor(mid_op, icR, icR'))
    tMPO[Nt] = replaceind(tMPO[Nt] * ten_mid_op, icR' => icR)
    @show inds(tMPO[Nt])

    for ii = Nt+1:b2
        tMPO[ii] = replaceinds(dag(Wc), (icP, icPs), (time_sites[ii]',time_sites[ii]) )  # TODO Check 
    end
    for ii = b2+1:Ntot
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(dag(Wc_im), (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end


    # Label linkinds
    # TODO phys ind of bl and tr must be first one here (in case thye're not product states)

    tMPO[1] = replaceinds(tMPO[1], (icL, icR), (ind(bl,1), time_links[1]))   

    for ii = 2:Ntot-1
        tMPO[ii] = replaceinds(tMPO[ii], (icL, icR), (time_links[ii-1],time_links[ii]))
    end

    tMPO[end] = replaceinds(tMPO[end], (icL, icR), (time_links[end] , Index(dim(icR), "Link,tr") ))

    # Contract boundary states (bottom/left)
    tMPO[1] = tMPO[1] * bl  

    return tMPO

end


function fwback_tMPO(ww::MPO, tr)
    tr_link = only(inds(ww[end],"Link,tr"))
    tr = to_itensor(tr, tr_link)
    ww[end] *= tr
    return ww
end


function fwback_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index}; tr, kwargs...)
    ww = fwback_tMPO_opentr(b, time_sites; kwargs...)
    fw_tMPO(ww, tr)
end



""" Builds forward tMPO with nbeta steps on one side only: 
in-U(β)-U(β)-..U(β)-U(idt)-U(idt)-U(idt)-U(idt)-fin 
   |___nbeta_____|     
   Returns tMPO 
"""
function fwback_tMPO_initbetaonly(b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl::ITensor = b.tp.bl, tr::ITensor)
    ww = fw_tMPO_opentr(b, time_sites; init_beta_only=true, bl, tr)
    fw_tMPO(ww, tr)
end



function fwback_left_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; kwargs...)
    fwback_tMPS(b,time_sites; LR=:left, kwargs...)
end
function fwback_right_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; kwargs...)
    fwback_tMPS(b,time_sites; LR=:right, kwargs...)
end


function fwback_tMPS(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fwback_tMPS(b, time_sites; kwargs...)
end

function fwback_tMPS(
    b::FwtMPOBlocks,
    time_sites::Vector{<:Index};
    bl = b.tp.bl,
    tr,
    LR::Symbol = :right,
    init_beta_only::Bool=false
)


    Ntot = length(time_sites) 
    @assert Ntot % 2 == 0 
    Nt = div(length(time_sites),2)


    bl = to_itensor(bl, "bl")
    tr = to_itensor(tr, "tr")
    tp = b.tp
    nbeta = tp.nbeta

    @assert nbeta <= Ntot

    # Choose direction-dependent fields and indices
    if LR == :left
        W = b.Wl
        W_im = b.Wl_im
        (iL, iR, iP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:Ps])
    elseif LR == :right
        W = b.Wr
        W_im = b.Wr_im
        (iL, iR, iP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P])
    else
        error("Unknown LR: $(LR)")
    end


    # Corner case length-1 tMPS 
    if Ntot == 1
        @assert nbeta == 0 # or we need to think more 
        A = replaceinds(W, (iL, iR, iP), (inds(bl)[1], inds(tr)[1], time_sites[1]))
        A *= bl 
        return MPS([A*tr])
    end



    # Make same indices for real and imag, it's easier afterwards 
    replaceinds!(W_im, inds(W_im), inds(W))

    b1,b2 = beta_lims(Ntot, nbeta, init_beta_only)

    rot_links_mps = [Index(dim(iL), "Link,rotl=$ii") for ii in 1:(Ntot - 1)]

    tMPS = MPS(Ntot)

    for ii = 1:b1
        #@info "$(ii) im" 
        tMPS[ii] = W_im * delta(iP, time_sites[ii])
    end
    for ii = b1+1:Nt
        #@info "$(ii) re" 
        tMPS[ii] = W * delta(iP, time_sites[ii])
    end
        for ii = Nt+1:b2
        #@info "$(ii) re" 
        tMPS[ii] = W * delta(iP, time_sites[ii])
    end
    for ii = b2+1:Ntot
        #@info "$(ii) im" 
        tMPS[ii] = dag(W_im) * delta(iP, time_sites[ii])
    end

    # Contract edges with boundary states, label linkinds
    tMPS[1] = tMPS[1] * bl * delta(ind(bl,1), iL) * delta(iR, rot_links_mps[1])

    for ii = 2:Ntot-1
        tMPS[ii] = tMPS[ii] * delta(iL, rot_links_mps[ii-1]) * delta(iR, rot_links_mps[ii])
    end

    tMPS[end] = (tMPS[end] * delta(iL, rot_links_mps[Ntot-1])) * (dag(tr) * delta(ind(tr,1), iR))

    return tMPS
end
