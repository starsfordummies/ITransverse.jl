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

