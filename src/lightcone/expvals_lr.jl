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
    psi_L = applys(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_ham_folded_tMPO(tp, opR, time_sites), 0, 1, "Site")
    psi_R = applys(tmpo, rr)

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    time_sites = siteinds(ll)
    tmpo = build_ham_folded_tMPO(tp,  fold_id, time_sites)
    psi_L = applys(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_ham_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi_R = applys(tmpo, rr)

    ev_L11R = overlap_noconj(psi_L,psi_R)

    return ev_LOOR/ev_L11R

end

""" TODO need to finish """
function expval_en_density(ll::MPS, rr::MPS, tp::tmpo_params)

    time_sites = siteinds(ll)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites)

    rho0 = (tp.init_state) * (tp.init_state')
    tMPO1[end] *= ITensor(rho0, ri1)
    tMPO2[end] *= ITensor(rho0, ri2)

    temp_s = siteinds("S=1/2",2)
    os = OpSum()
    os += tp.mp.JXX, "X",1,"X",2
    os += tp.mp.hz,  "I",1,"Z",2
    os += tp.mp.hz,  "Z",1,"I",2
    os += tp.mp.λx,  "I",1,"X",2
    os += tp.mp.λx,  "X",1,"I",2

    #ϵ_op = ITensor(os, temp_s, temp_s')
    ϵ_op = MPO(os, temp_s)
    cs1 = combiner(temp_s[1], temp_s[1]')
    cs2 = combiner(temp_s[2], temp_s[2]')
    ϵ_op[1] *= cs1 
    ϵ_op[2] *= cs2 
    ϵ_op[1] *= delta(combinedind(cs1), li1)
    ϵ_op[2] *= delta(combinedind(cs2), li2)

    tMPO_eps = applys(tMPO2, tMPO1)
    tMPO_eps[1] *= delta(li2',li2)
    tMPO_eps[1] *= ϵ_op[1]
    tMPO_eps[1] *= ϵ_op[2]

    #@show inds(tMPO_eps[1])

    tMPO_ids = applys(tMPO2, tMPO1)
    tMPO_ids[1] *= delta(li2',li2)

    tMPO_ids[1] *= ITensor(ComplexF64[1,0,0,1], li1)
    tMPO_ids[1] *= ITensor(ComplexF64[1,0,0,1], li2)

    #normalization 
    LOO = apply(tMPO_ids, ll)
    ev_L11R = overlap_noconj(LOO, rr)

    LOO = apply(tMPO_eps, ll)
    ev_LOOR = overlap_noconj(LOO, rr)

    return ev_LOOR/ev_L11R

end


function compute_expvals(ll::AbstractMPS, rr::AbstractMPS, op_list::Vector{String}, tp::tmpo_params)

    if op_list[1] == "all"
        op_list = ["X", "Z", "XX", "ZZ", "eps"]
    end

    allevs = Dict()

    for op in op_list
        if op == "X"
            println("X")
            allevs["X"] = expval_LR(ll, rr, ComplexF64[0,1,1,0], tp)
        elseif op == "Z"
            println("Z")
            allevs["Z"] = expval_LR(ll, rr, ComplexF64[1,0,0,-1], tp)
        elseif op == "eps"
            println("eps")
            allevs["eps"] = expval_en_density(ll, rr, tp)
        end
    end

    return allevs
end
