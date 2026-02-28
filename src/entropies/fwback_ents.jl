""" TM chunk builder for symmetric case. If the chunk is in the middle of the MPS, it has 4 open legs
so expensive to store. Allows to start from a `prev_chunk` ITensor. 
If `flip_fb=true`, it unfolds and partial transposes before contracting physical legs,
so contracts fw leg with backward legs (useful for building rho2 with `psi` seen as fw-vs-back TM) """
function tm_chunk(psi::MPS, i1::Int, i2::Int, prev_chunk::ITensor=ITensor(1); flip_fb::Bool, contract_from_right::Bool=false)
    tm = prev_chunk
    ss = siteinds(psi)
    psi2 = prime(linkinds, psi)
    rrange = contract_from_right ? reverse(i1:i2) : (i1:i2) 
    for kk in rrange
        tm *= psi[kk]
        if flip_fb
            tm = ptranspose_contract(tm, psi2[kk], ss[kk], ss[kk])  # or: ptranspose_contract()
        else
            tm *= psi2[kk]
        end
        @assert ndims(tm) < 5 "?? $(inds(tm))"
    end

    return tm
end





""" Takes an input MPS and returns a chunk of it partial-traced over its folded indices between `n1` and `n2` """
function ptr_chunk(psi::MPS, n1::Int, n2::Int; contract_from_right::Bool=false)
    ss = siteinds(psi)

    chunk = ITensor(1)
    rrange = contract_from_right ? reverse(n1:n2) : n1:n2
    for kk in rrange
        chunk *= trace_combinedind(psi[kk], ss[kk])
    end

    @assert ndims(chunk) < 3 "?? $(inds(chunk))"

    return chunk
end


""" Given input a folded `psi`, we reopen its legs to view it as a fw-back density matrix `rho`,
then compute its purity tr_A(rho^2) for a bipartition A=1:cut, B=cut+1:N """
function rho2_fwback(psi::MPS, cut::Int)

    LL = length(psi)

    # Normalization: tr(rho) = 1 
    tr_rho = scalar(ptr_chunk(psi, 1, LL))

    # tr_B
    blockB = ptr_chunk(psi, cut+1, LL; contract_from_right=true)
  
    rho = tm_chunk(psi, 1, cut; flip_fb=true)
    
    rho *= blockB 
    rho *= prime(blockB)

    return scalar(rho)/(tr_rho^2)

end


""" Given input a folded `psi`, we reopen its legs to view it as a fw-back density matrix `rho`,
then compute its purity tr_A(rho^4) for a bipartition A=1:cut, B=cut+1:N """
function rho4_fwback(psi::MPS, cut::Int; alg="densitymatrix", cutoff=1e-12, maxdim=maxlinkdim(psi))

    LL = length(psi)

    # Normalization: tr(rho) = 1 
    tr_rho = scalar(ptr_chunk(psi, 1, LL))

    psit = if cut < LL
        # tr_B
        blockB = ptr_chunk(psi, cut+1, LL, contract_from_right=true)

        psimats = psi[1:cut]
        psimats[end] *= blockB
        MPS(psimats)
    else
        psi 
    end

    psi = reopen_inds(psit;  different_fwback_inds=false)

    psit2 = contract(psit, psit'; alg, cutoff, maxdim)
    psit2 = join_inds(psit2)

    rho = tm_chunk(psit2, 1, cut; flip_fb=true)
    
    return scalar(rho)/(tr_rho^4)

end


""" Given input a folded `psi`, we reopen its legs to view it as a fw-back density matrix `rho`,
then compute its purity tr(rho^2) for a segment """
function rho2_fwback_segment(psi::MPS, iA::Int, fA::Int, iB::Int, fB::Int)

    LL = length(psi)

    # Normalization: tr(rho) = 1 
    tr_rho = scalar(ptr_chunk(psi::MPS, 1, LL))

    #AB 
    block = ptr_chunk(psi::MPS, 1, iA-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iA, fA, rhoL; flip_fb=true)
    

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fA+1, iB-1)
    rhoL *= block 
    rhoL *= prime(block)

    rhoL = tm_chunk(psi::MPS, iB, fB, rhoL; flip_fb=true)
    @assert ndims(rhoL)< 3

    block = ptr_chunk(psi::MPS, fB+1, length(psi))
    rhoL *= block 
    rhoL *= prime(block)

    rhoAB = scalar(rhoL)/(tr_rho^2)


    #A 
    block = ptr_chunk(psi::MPS, 1, iA-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iA, fA, rhoL; flip_fb=true)

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fA+1, LL)
    rhoL *= block 
    rhoL *= prime(block)

    rhoA = scalar(rhoL)/(tr_rho^2)

    #B 
    block = ptr_chunk(psi::MPS, 1, iB-1)
    rhoL = block * prime(block)

    rhoL = tm_chunk(psi::MPS, iB, fB, rhoL; flip_fb=true)

    @assert ndims(rhoL) < 3
    block = ptr_chunk(psi::MPS, fB+1, LL)
    rhoL *= block 
    rhoL *= prime(block)

    rhoB = scalar(rhoL)/(tr_rho^2)

    return rhoA, rhoB, rhoAB
end



function compute_rho2s(psi::MPS, length_intervals::Int)
    LL = length(psi)
    mid = div(LL+1,2)


    rhos_lr = []
    rhos_left = []
    for distance=0:div(LL-2*length_intervals,2)-1
        iA = mid-distance-length_intervals
        fA = mid-distance
        iB = mid+distance+1
        fB = mid+distance+length_intervals

        @info "computing for [$(iA)-$(fA)]-[$(iB)-$(fB)]"
        #r2s_LR = rho2_lr(psi::MPS, iA, fA, iB, fB)

        r2s_left = rho2_left(psi::MPS, iA, fA, iB, fB)


        #push!(rhos_lr, r2s_LR)
        push!(rhos_left, r2s_left)
    end

    return rhos_lr, rhos_left

end

function compute_sn_cut(psi::MPS, n::Int; cut::Int=div(length(psi),2), cutoff=1e-10, maxdim=maxlinkdim(psi))
    ss = siteinds(psi)

    for kk = length(psi):-1:cut
        psi[kk] = trace_combinedind(psi[kk], ss[kk])
    end

    psi = ITenUtils.contract_dangling!(psi)

    orthogonalize!(psi,1)

    @show length(psi)

    oo = reopen_inds!(psi, different_fwback_inds=false)

    oo = oo/ITenUtils.trace_mpo(oo)

    @show siteinds(oo)
    @show linkinds(oo)

    rho2 = copy(oo)
    for kk = 2:n
        @show kk
        rho2 = apply(oo, rho2; cutoff,maxdim)
    end

    ITenUtils.trace_mpo(rho2)
end


function t4mid_slice(psi::MPS)
    NN = length(psi)
    cut = div(NN,2)


    ll = linkinds(psi)
    phi = prime(linkinds,psi)

    for ii = 1:dim(ll[cut])
        for jj = 1:dim(ll[cut])
                
            lenv = ITensors.OneITensor()
            for kk = 1:4
                lenv *= psi[kk]
                lenv *= phi[kk]
            end

        end

    end

    return 
end



function trrho_lr(psi::MPS, iA::Int, fA::Int, iB::Int=length(psi)+1, fB::Int=length(psi)+1)

    LL = length(psi)
    ss = siteinds(psi)

    psip = prime(linkinds, psi)

    trrho = ITensor(1.)

    for kk = 1:iA-1
        trrho *= psi[kk] 
        trrho *= psip[kk]
    end
    for kk = iA:fA 
        trrho *= trace_combinedind(psi[kk], ss[kk])
        trrho *= trace_combinedind(psip[kk], ss[kk])
    end
    for kk = fA+1:iB-1
        trrho *= psi[kk] 
        trrho *= psip[kk]
    end

    if iB <= LL
        for kk = iB:fB
            trrho *= trace_combinedind(psi[kk], ss[kk]) 
            trrho *= trace_combinedind(psip[kk], ss[kk])
        end
        for kk = fB+1:LL
            trrho *= psi[kk] 
            trrho *= psip[kk]
        end
    end
    
    return scalar(trrho)
end
