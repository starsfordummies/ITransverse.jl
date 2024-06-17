""" Build exp value <L|O|R> for a single operator `op` """
function expval_LR(ll::MPS, rr::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = build_ham_folded_tMPO(tp, fold_id, time_sites)
    psi1L = applys(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_ham_folded_tMPO(tp, op, time_sites), 0, 1, "Site")
    psiOR = applys(tmpo, rr)

    tmpo = swapprime(build_ham_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi1R = applys(tmpo, rr)


    L1OR = overlap_noconj(psi1L,psiOR)
    L11R = overlap_noconj(psi1L,psi1R)

    return L1OR/L11R

end

# Version with 2 operators
function expval_LR(ll::MPS, rr::MPS, opL::Vector{ComplexF64}, opR::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = build_ham_folded_tMPO(tp,  opL, time_sites)
    psi_L = apply(tmpo, ll)

    tmpo = build_ham_folded_tMPO(tp, fold_id, time_sites)
    psi_L_id = apply(tmpo, rr)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_ham_folded_tMPO(tp, opR, time_sites), 0, 1, "Site")
    psi_R = apply(tmpo, rr)

    tmpo = swapprime(build_ham_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi_R_id = apply(tmpo, rr)


    ev = overlap_noconj(psi_L,psi_R)/overlap_noconj(psi_L_id,psi_R_id)

    return ev

end

""" TODO need to finish """
function expval_en_density(ll::MPS, rr::MPS, tp::tmpo_params)

    time_sites = siteinds(ll)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites)

    rho0 = ITensor(tp.init_state) * (tp.init_state')
    tMPO1[end] *= ITensor(rho0, ri1)
    tMPO2[end] *= ITensor(rho0, ri2)

    os = OpSum()
    os += tp.mp.JXX, "X",1,"X",2
    os += tp.mp.hz,  "I",1,"Z",2
    os += tp.mp.hz,  "Z",1,"I",2
    os += tp.mp.λx,  "I",1,"X",2
    os += tp.mp.λx,  "X",1,"I",2

    ϵ_op = ITensor(op, li1, li2')

    tMPO_eps = tMPO1 * tMPO2' 
    tMPO_eps[1] *= ϵ_op


end

