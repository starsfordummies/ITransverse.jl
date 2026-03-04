""" Partial traces the input `psi` seen as a fw-back density matrix
in the intervals iA:fA and iB:fB """
function trrho_fwback(psi::MPS, iA::Int, fA::Int, iB::Int=length(psi)+1, fB::Int=length(psi)+1)

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


""" Computes left-right RDM purities for symmetric case L=R """
function mutuals_fwback_segment(psi::MPS, iA::Int, fA::Int, iB::Int, fB::Int)

    LL = length(psi)
    ss = siteinds(psi)

    trrho = scalar(ptr_chunk(psi, 1, LL))

    psip = prime(linkinds,psi)

    left_block = ptr_chunk(psi, 1, iA-1) 
    rhoAB = left_block * left_block'
    for kk = iA:fA
        rhoAB *= psi[kk]
        rhoAB = ptranspose_contract(rhoAB, psip[kk], ss[kk])
    end

    center_right_block = ptr_chunk(psi, fA+1, LL; contract_from_right=true)
    rhoA = rhoAB * center_right_block
    rhoA = rhoA * center_right_block'

    center_block = ptr_chunk(psi, fA+1, iB-1)
    rhoAB *= center_block
    rhoAB *= center_block'
  
    for kk = iB:fB
        rhoAB *= psi[kk]
        rhoAB = ptranspose_contract(rhoAB, psip[kk], ss[kk])
    end
    right_block = ptr_chunk(psi, fB+1, LL; contract_from_right=true)

    rhoAB *= right_block
    rhoAB *= right_block'

    rhoB = right_block * right_block'
     for kk = reverse(iB:fB)
        rhoB *= psi[kk]
        rhoB = ptranspose_contract(rhoB, psip[kk], ss[kk])
    end
    left_center_block = ptr_chunk(psi, 1, iB-1)

    rhoB *= left_center_block
    rhoB *= left_center_block'


    return scalar(rhoAB)/trrho^2, scalar(rhoA)/trrho^2, scalar(rhoB)/trrho^2
end


""" Computes left-right RDM purities for symmetric case L=R """
function mutuals_lr_TODO_segment(psi::MPS, iA::Int, fA::Int, iB::Int, fB::Int)


    blockL = tm_chunk(psi::MPS, 1, iA-1; flip_fb=false)
    blockLL = tm_chunk(psi::MPS, iA, iB-1, blockL; flip_fb=false)

    @assert ndims(blockLL) < 3 "?? $(inds(blockLL))"

    blockR = tm_chunk(psi::MPS, fB+1, length(psi),; flip_fb=false, contract_from_right=true)
    blockRR = tm_chunk(psi::MPS, fA+1, fB, blockR; flip_fb=false, contract_from_right=true)

    @assert ndims(blockRR) < 3 "?? $(inds(blockRR))"

    # Normalization(s?): tr(rho) = 1 

    trrho_A = trrho_lr(psi, iA, fA)
    trrho_B = trrho_lr(psi, iB, fB)
    trrho_AB = trrho_lr(psi, iA, fA, iB, fB)




    tr_block = tm_chunk(psi::MPS, iA, fA; flip_fb=true)

    rhoLR = blockL
    rhoLR *= prime(blockL,2)

    rhoLR *= prime(tr_block)
    rhoLR *= swapprime(tr_block, 1 => 3)

    @assert ndims(rhoLR) < 5

    @show inds(blockRR)


    rhoLR_A = scalar((rhoLR * blockRR) *  prime(blockRR,2))/(trrho_A^2)


    mid_block = tm_chunk(psi::MPS, fA+1, iB-1; flip_fb=false)
    rhoLR = rhoLR * mid_block 
    rhoLR *= prime(mid_block,2)


    tr_block = tm_chunk(psi::MPS, iB, fB; flip_fb=true, contract_from_right=true)
    rhoLR *= prime(tr_block)
    rhoLR *= swapprime(tr_block, 1 => 3)
    @assert ndims(rhoLR)< 5

    rhoLR *= blockR 
    rhoLR *= prime(blockR,2)

    rhoLR = scalar(rhoLR)/(trrho_AB^2)

    rhoLR_B = blockR
    rhoLR_B *= prime(blockR,2)

    rhoLR_B *= prime(tr_block)
    rhoLR_B *= swapprime(tr_block, 1 => 3)

    rhoLR_B = scalar((rhoLR_B * blockLL) *  prime(blockLL,2))/(trrho_B^2)

    return rhoLR_A, rhoLR_B, rhoLR
end
