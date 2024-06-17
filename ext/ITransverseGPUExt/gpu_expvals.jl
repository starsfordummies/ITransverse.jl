""" Take as input two CUDA MPS, computes <L|O|R> in the GPU"""
function ITransverse.gpu_expval_LR(ll::MPS, rr::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = NDTensors.cu(build_ham_folded_tMPO(tp,  fold_id, time_sites))
    psi_L = applys(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = NDTensors.cu(swapprime(build_ham_folded_tMPO(tp, op, time_sites), 0, 1, "Site"))
    psi_R = applys(tmpo, rr)

    ev_LOR = overlap_noconj(psi_L,psi_R)

    tmpo = NDTensors.cu(swapprime(build_ham_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site"))
    psi_R = applys(tmpo, rr)

    ev_L1R = overlap_noconj(psi_L,psi_R)
    ev = ev_LOR/ev_L1R

    return ev

end

""" Take as input two CUDA MPS, computes <L|O|R> on the CPU """
function ITransverse.cpu_expval_LR(ll::MPS, rr::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    llc = NDTensors.cpu(ll)
    rrc = NDTensors.cpu(rr)
    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(llc)
    tmpo = build_ham_folded_tMPO(tp, fold_id, time_sites)
    psi_L = applys(tmpo, llc)

    time_sites = siteinds(rrc)
    tmpo = swapprime(build_ham_folded_tMPO(tp, op, time_sites), 0, 1, "Site")
    psi_R = applys(tmpo, rrc)

    ev_LOR = overlap_noconj(psi_L,psi_R)

    tmpo = swapprime(build_ham_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi_R = applys(tmpo, rrc)

    ev_L1R = overlap_noconj(psi_L,psi_R)
    ev = ev_LOR/ev_L1R

    return ev

end



function gpu_expval_LL_sym(ll::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = NDTensors.cu(build_ham_folded_tMPO(tp,  fold_id, time_sites))
    psi_L = applys(tmpo, ll)

    tmpo = NDTensors.cu(swapprime(build_ham_folded_tMPO(tp, op, time_sites), 0, 1, "Site"))
    psi_R = applys(tmpo, ll)

    ev_LOR = overlap_noconj(psi_L,psi_R)
    ev_L1R = overlap_noconj(psi_L,psi_L)

    ev = ev_LOR/ev_L1R

    return ev

end
