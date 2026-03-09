using ITensors, ITensorMPS, ITransverse
using ITensors.Adapt
using ITransverse.ITenUtils: to_itensor
using Test

""" Unfolded tMPO with 
- `nbetai` initial steps of imaginary time evolution 
- `nfw` steps of forward time evolution 
-  (optionally) a `mid_op` operator insertion 
- `nback` steps of backwards time evolution
- `nbetaf` steps of imaginary time evolution
"""
function fwback_tMPO_old(b::FwtMPOBlocks, time_sites::Vector{<:Index}, nbetai::Int, nfw::Int, nback::Int, nbetaf::Int; 
    mid_op = [1,0,0,1], t_op::Int=nbetai+nfw, bl::ITensor = b.tp.bl, tr = b.tp.bl)

    @info "Building fwback with $(nbetai)-$(nfw)-$(nback)-$(nbetaf) - operator at $(nbetai+nfw)"

    Ntot = length(time_sites) 
    @assert nbetai + nfw + nback + nbetaf == Ntot

    (; tp, Wc, Wc_im, rot_inds) = b
    (icL, icR, icP, icPs) = (rot_inds[:L], rot_inds[:R], rot_inds[:P], rot_inds[:Ps]) 

    elt = NDTensors.unwrap_array_type(tp.bl)

    tr = adapt(elt, to_itensor(tr, icR))
    ind_op = sim(icR, tags="op")
    ten_mid_op = adapt(elt, ITensor(mid_op, ind_op, ind_op'))

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    time_links = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Ntot-1)]

    tMPO =  MPO(Ntot)

    for ii = 1:nbetai
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(Wc_im, (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end
    for ii = nbetai+1:nbetai+nfw
        tMPO[ii] = replaceinds(Wc, (icP, icPs), (time_sites[ii],time_sites[ii]') )
    end

    for ii = nbetai+nfw+1:nbetai+nfw+nback
        tMPO[ii] = replaceinds(dag(Wc), (icP, icPs), (time_sites[ii]',time_sites[ii]) )  # TODO Check 
    end
    for ii = nbetai+nfw+nback+1:Ntot
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(dag(Wc_im), (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end


    # Label linkinds
    # TODO phys ind of bl and tr must be first one here (in case thye're not product states)

    tMPO[1] = replaceinds(tMPO[1], (icL, icR), (ind(bl,1), time_links[1]))   

    for ii = 2:Ntot-1
        tMPO[ii] = replaceinds(tMPO[ii], (icL, icR), (time_links[ii-1],time_links[ii]))
    end

    tMPO[end] = replaceind(tMPO[end], icL => time_links[end])


    # Plug operator in the column
    cl = commonind(tMPO[t_op], tMPO[t_op+1])
    tMPO[t_op] = replaceind(contract(tMPO[t_op], ten_mid_op, cl, ind_op), ind_op' => cl)
    #@show inds(tMPO[Nt])
    
    # Contract boundary states (bottom/left and top/right)
    tMPO[1] = tMPO[1] * bl  
    tMPO[end] = tMPO[end] * tr  

    return tMPO

end

@testset "new fwmpo builder" begin

JXX = 1.0
hz = 0.7
gx = 0.0
#H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X


dt = 0.1

# init_state = plus_state
init_state = up_state

mp = IsingParams(JXX, hz, gx)

tp = tMPOParams(dt,  expH_ising_murg, mp, 0, init_state)
b = FwtMPOBlocks(tp)

ss = siteinds("S=1/2", 16)

tmpo_new = fwback_tMPO(b, ss, 2, 6, 6, 2, mid_op = [1,0,0,-1], tr=b.tp.bl)
tmpo_old = fwback_tMPO_old(b, ss, 2, 6, 6, 2, mid_op = [1,0,0,-1], tr=b.tp.bl)

@test tmpo_new ≈ tmpo_old

end
