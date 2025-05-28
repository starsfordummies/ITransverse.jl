""" Builds temporal MPO and starting tMPS guess for *forward evolution only* 
with `nbeta` steps of imaginary time regularization.
Closes with initial state again, so it's a Loschmidt echo type setup.

Returns (tMPO, tMPS) pair.

The structure built (Loschmidt style) after rotation is
```
(left[bottom]_state)---Wβ--Wβ---Wt-Wt-Wt-...-Wt---Wβ--Wβ---(right[top]_state)
                      (nbeta)                     (nbeta)
```
 """

function fw_tMPO(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)
    b = FwtMPOBlocks(tp)
    fw_tMPO(b, time_sites; kwargs...)
end


function fw_tMPO(b::FwtMPOBlocks, time_sites::Vector{<:Index};  bl::ITensor = b.tp.bl, tr)

    tr = to_itensor(tr, "tr")

    (; tp, Wc, Wc_im, rot_inds) = b

    nbeta = tp.nbeta 

    @assert nbeta == 0 || nbeta < length(time_sites) - 2

    Nsteps = length(time_sites)

    (icL, icR, icP, icPs) = (rot_inds[:L], rot_inds[:R], rot_inds[:P], rot_inds[:Ps]) 

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    time_links = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]


    tMPO =  MPO(fill(Wc, Nsteps))

    for ii = 1:nbeta
        tMPO[ii] = replaceinds(Wc_im, (icP, icPs), (time_sites[ii],time_sites[ii]'))
        #(Wc_im) * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]')
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPO[ii] = replaceinds(Wc, (icP, icPs), (time_sites[ii],time_sites[ii]') )
        #tMPO[ii] = tMPO[ii] * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]') 

    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPO[ii] = replaceinds(dag(Wc_im), (icP, icPs), (time_sites[ii],time_sites[ii]'))
        #tMPO[ii] = dag(Wc_im) * delta(icPs, time_sites[ii]') * delta(icP, time_sites[ii]) 
    end


    # Contract edges with boundary states, label linkinds
    # TODO phys ind of bl and tr must be first one here (in case thye're not product states)

    tMPO[1] = replaceinds(tMPO[1], (icL, icR), (ind(bl,1), time_links[1]))   
    tMPO[1] = tMPO[1] * bl  

    for ii = 2:Nsteps-1
        #tMPO[ii] = tMPO[ii] * delta(icL, time_links[ii-1]) * delta(icR, time_links[ii]) 
        tMPO[ii] = replaceinds(tMPO[ii], (icL, icR), (time_links[ii-1],time_links[ii]))
    end

    tMPO[end] = replaceinds(tMPO[end], (icL, icR), (time_links[end], ind(tr,1)))
    tMPO[end] = tMPO[end] * dag(tr)

    return tMPO

end


function fw_tMPOtMPS(b::FwtMPOBlocks, time_sites::Vector{<:Index};  bl::ITensor = b.tp.bl, tr)

    tr = to_itensor(tr, "tr")

    tp = b.tp

    nbeta = tp.nbeta 

    @assert nbeta < length(time_sites) - 2

    Nsteps = length(time_sites)

    # Rotated indices already 
    Wc = b.Wc

    (icL, icR, icP, icPs) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P], b.rot_inds[:Ps]) 

    Wc_im = b.Wc_im

    Wr = b.Wr
    (irL, irR, irP) =  (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P]) 

    Wr_im = b.Wr_im

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))
    replaceinds!(Wr_im, inds(Wr_im), inds(Wr))

    rot_links_mpo = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]
    rot_links_mps = sim(rot_links_mpo)


    tMPO =  MPO(fill(Wc, Nsteps))
    tMPS =  MPS(fill(Wr, Nsteps))

    for ii = 1:nbeta
        tMPO[ii] = (Wc_im) * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]')
        tMPS[ii] = (Wr_im) * delta(irP, time_sites[ii]) 
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPO[ii] = tMPO[ii] * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]') 
        tMPS[ii] = tMPS[ii] * delta(irP, time_sites[ii])
    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPO[ii] = dag(Wc_im) * delta(icPs, time_sites[ii]') * delta(icP, time_sites[ii]) 
        tMPS[ii] = dag(Wr_im) * delta(irP, time_sites[ii]) 
    end


    # Contract edges with boundary states, label linkinds
    tMPO[1] = tMPO[1] * bl * delta(ind(bl,1), icL) * delta(icR, rot_links_mpo[1]) 
    tMPS[1] = tMPS[1] * bl * delta(ind(bl,1), irL) * delta(irR, rot_links_mps[1]) 

    for ii = 2:Nsteps-1
        tMPO[ii] = tMPO[ii] * delta(icL, rot_links_mpo[ii-1]) * delta(icR, rot_links_mpo[ii]) 
        tMPS[ii] = tMPS[ii] * delta(irL, rot_links_mps[ii-1]) * delta(irR, rot_links_mps[ii]) 
    end

    tMPO[end] = (tMPO[end] * delta(icL, rot_links_mpo[Nsteps-1])) * (dag(tr) * delta(ind(tr,1),icR))
    tMPS[end] = (tMPS[end] * delta(irL, rot_links_mps[Nsteps-1])) * (dag(tr) * delta(ind(tr,1),irR))

    return tMPO, tMPS

end


""" Builds forward tMPO with nbeta steps on one side only: 
in-U(β)-U(β)-..U(β)-U(idt)-U(idt)-U(idt)-U(idt)-fin 
   |___nbeta_____|     
   Returns tMPO and tMPS     
"""
function fw_tMPO_initbetaonly(b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl::ITensor = b.tp.bl, tr::ITensor)

    tp = b.tp

    # CPU or GPU ?
    dttype = NDTensors.unwrap_array_type(b.Wc)
    bl = adapt(dttype, bl)
    right_state = adapt(dttype, tr)

    nbeta = tp.nbeta

    @assert nbeta < length(time_sites) - 1

    Nsteps = length(time_sites)

    # Rotated indices already 
    Wc = b.Wc
    (icL, icR, icP, icPs) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P], b.rot_inds[:Ps]) 
    Wc_im = b.Wc_im

    Wr = b.Wr
    (irL, irR, irP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P]) 

    Wr_im = b.Wr_im

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))
    replaceinds!(Wr_im, inds(Wr_im), inds(Wr))

    rot_links_mpo = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]
    rot_links_mps = sim(rot_links_mpo)


    tMPO =  MPO(Nsteps)
    tMPS =  MPS(Nsteps)

    for ii = 1:nbeta
        tMPO[ii] = (Wc_im) * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]')
        tMPS[ii] = (Wr_im) * delta(irP, time_sites[ii]) 
    end
    for ii = nbeta+1:Nsteps
        tMPO[ii] = Wc * delta(icP, time_sites[ii]) * delta(icPs, time_sites[ii]') 
        tMPS[ii] = Wr * delta(irP, time_sites[ii])
    end


    # Contract edges with boundary states, label linkinds
    tMPO[1] = tMPO[1] * bl * delta(ind(bl,1), icL) * delta(icR, rot_links_mpo[1]) 
    tMPS[1] = tMPS[1] * bl * delta(ind(bl,1), irL) * delta(irR, rot_links_mps[1]) 

    for ii = 2:Nsteps-1
        tMPO[ii] = tMPO[ii] * delta(icL, rot_links_mpo[ii-1]) * delta(icR, rot_links_mpo[ii]) 
        tMPS[ii] = tMPS[ii] * delta(irL, rot_links_mps[ii-1]) * delta(irR, rot_links_mps[ii]) 
    end

    tMPO[end] = (tMPO[end] * delta(icL, rot_links_mpo[Nsteps-1])) * (dag(tr) * delta(ind(tr,1),icR))
    tMPS[end] = tMPS[end] * delta(irL, rot_links_mps[Nsteps-1]) * (dag(tr) * delta(ind(tr,1),irR))

    return tMPO, tMPS

end


""" For quick debugging, build a forward tMPO for non-integrable Ising with random params """
function rand_ising_fwtmpo(time_sites=siteinds("S=1/2",20))

    plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])
    mp = IsingParams(1.0, -rand(), rand())
    tp = tMPOParams(0.1, build_expH_ising_murg, mp, 0, plus_state)
    b = FwtMPOBlocks(tp)

    ww, _ = fw_tMPO(b, time_sites; tr=plus_state)

    return ww
end



function fw_left_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl = b.tp.bl, tr)
    fw_tMPS(b,time_sites;bl,tr, LR=:left)
end
function fw_right_tMPS( b::FwtMPOBlocks, time_sites::Vector{<:Index}; bl = b.tp.bl, tr)
    fw_tMPS(b,time_sites;bl,tr, LR=:right)
end

function fw_tMPS(
    b::FwtMPOBlocks,
    time_sites::Vector{<:Index};
    bl = b.tp.bl,
    tr,
    LR::Symbol = :left
)

    bl = to_itensor(bl, "bl")
    tr = to_itensor(tr, "tr")
    tp = b.tp
    nbeta = tp.nbeta
    @assert nbeta == 0 || nbeta < length(time_sites) - 2
    Nsteps = length(time_sites)

    # Choose direction-dependent fields and indices
    if LR == :left
        W = b.Wl
        W_im = b.Wl_im
        (iL, iR, iP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P])
    elseif LR == :right
        W = b.Wr
        W_im = b.Wr_im
        (iL, iR, iP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P])
    else
        error("Unknown LR: $(LR)")
    end

    # Make same indices for real and imag, it's easier afterwards 
    replaceinds!(W_im, inds(W_im), inds(W))

    rot_links_mps = [Index(dim(iL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]

    tMPS = MPS(fill(W, Nsteps))

    for ii = 1:nbeta
        tMPS[ii] = W_im * delta(iP, time_sites[ii])
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPS[ii] = tMPS[ii] * delta(iP, time_sites[ii])
    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPS[ii] = dag(W_im) * delta(iP, time_sites[ii])
    end

    # Contract edges with boundary states, label linkinds
    tMPS[1] = tMPS[1] * bl * delta(ind(bl,1), iL) * delta(iR, rot_links_mps[1])

    for ii = 2:Nsteps-1
        tMPS[ii] = tMPS[ii] * delta(iL, rot_links_mps[ii-1]) * delta(iR, rot_links_mps[ii])
    end

    tMPS[end] = (tMPS[end] * delta(iL, rot_links_mps[Nsteps-1])) * (dag(tr) * delta(ind(tr,1), iR))

    return tMPS
end
