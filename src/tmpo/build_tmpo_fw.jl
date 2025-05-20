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


# Alternate versions using building blocks


function fw_tMPO(tp::tMPOParams, time_sites::Vector{<:Index}; kwargs...)

    b = FwtMPOBlocks(tp)

    tpim = tMPOParams(tp; dt=-im*tp.dt)

    b_im = FwtMPOBlocks(tpim)

    fw_tMPO(b, b_im, time_sites; kwargs...)
end


function fw_tMPO(b::FwtMPOBlocks, b_im::FwtMPOBlocks, time_sites::Vector{<:Index}; 
    bl::ITensor = b.tp.bl, tr)

    tp = b.tp

    nbeta = tp.nbeta 

    @assert nbeta < length(time_sites) - 2

    Nsteps = length(time_sites)

    # Rotated indices already 
    Wc = b.Wc

    (icL, icR, icP, icPs) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P], b.rot_inds[:Ps]) 

    Wc_im = b_im.Wc

    Wr = b.Wr
    (irL, irR, irP) =  (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P]) 

    Wr_im = b_im.Wr

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
function fw_tMPO_initbetaonly(b::FwtMPOBlocks, b_im::FwtMPOBlocks, time_sites::Vector{<:Index}; 
    bl::ITensor = b.tp.bl, tr)

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
    Wc_im = b_im.Wc

    Wr = b.Wr
    (irL, irR, irP) = (b.rot_inds[:L], b.rot_inds[:R], b.rot_inds[:P]) 

    Wr_im = b_im.Wr

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

    ww, _ = fw_tMPO(b, b, time_sites; tr=plus_state)

    return ww
end
