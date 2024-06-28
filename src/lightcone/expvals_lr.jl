""" Build exp value <L|O|R> for a single operator `op`.
 In order to avoid normalization issues, we build both <L|OR> and <L|1R> overlaps, 
 the exp value is given by <L|OR>/<L|1R> """
function expval_LR(ll::MPS, rr::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(rr)
    tmpo = swapprime(build_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi1R = applyn(tmpo, rr)

    L1R = overlap_noconj(ll,psi1R)

    tmpo = swapprime(build_folded_tMPO(tp, op, time_sites), 0, 1, "Site")
    psiOR = applyn(tmpo, rr)

    LOR = overlap_noconj(ll,psiOR)



    return LOR/L1R

end


""" Build exp value <L|O|R> for a single operator `op` 
in a symmetric way, by building both <L1|OR> and <L1|1R> overlaps, 
 the exp value is given by <L1|OR>/<L1|1R>. 
 This should really be equivalent to `expval_LR`, but could be useful for checks if 
 we are not sure whether our L and R vectors are well converged. """
function expval_LR_twocol(ll::MPS, rr::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = build_folded_tMPO(tp, fold_id, time_sites)
    psi1L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_folded_tMPO(tp, op, time_sites), 0, 1, "Site")
    psiOR = applyn(tmpo, rr)

    tmpo = swapprime(build_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi1R = applyn(tmpo, rr)


    L1OR = overlap_noconj(psi1L,psiOR)
    L11R = overlap_noconj(psi1L,psi1R)

    return L1OR/L11R

end

""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` """ 
function expval_LR(ll::MPS, rr::MPS, opL::Vector{ComplexF64}, opR::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = build_folded_tMPO(tp,  opL, time_sites)
    psi_L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_folded_tMPO(tp, opR, time_sites), 0, 1, "Site")
    psi_R = applyn(tmpo, rr)

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    time_sites = siteinds(ll)
    tmpo = build_folded_tMPO(tp,  fold_id, time_sites)
    psi_L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_folded_tMPO(tp, fold_id, time_sites), 0, 1, "Site")
    psi_R = applyn(tmpo, rr)

    ev_L11R = overlap_noconj(psi_L,psi_R)

    return ev_LOOR/ev_L11R

end


""" Alternative way of computing expval_LR using open tMPOs and closing them. 
We pass the list of local operators to compute as a (regular) MPO, 
which we contract to the top of the tMPO 
It may be more flexible """
function expval_LR_open(ll::MPS, rr::MPS, ops::MPO, tp::tmpo_params)

    # TODO allow for longer MPOs (longer lists of local ops)

    @assert length(ops) == 2

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites_L)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites_R)

    rho0 = (tp.init_state) * (tp.init_state')
    tMPO1[end] *= ITensor(rho0, ri1)
    tMPO2[end] *= ITensor(rho0, ri2)
 
    ϵ_op[1] *= delta(combinedind(cs1), li1)
    ϵ_op[2] *= delta(combinedind(cs2), li2)

    LO = applyn(tMPO1, ll)
    OR = applyn(tMPO2, rr) # todo swap indices for non-symmetric MPOs

    insert!(LO.data, 1, ϵ_op[1])
    insert!(OR.data, 1, ϵ_op[2])

    ev_LOOR = overlap_noconj(LO, OR)

    deleteat!(LO.data,1)
    deleteat!(OR.data,1)

    LO[1] *= ITensor(ComplexF64[1,0,0,1], li1)
    OR[1] *= ITensor(ComplexF64[1,0,0,1], li2)

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end



""" Compute exp value for energy density in Ising """
function expval_en_density(ll::MPS, rr::MPS, tp::tmpo_params)

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites_L)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites_R)

    rho0 = (tp.init_state) * (tp.init_state')
    tMPO1[end] *= ITensor(rho0, ri1)
    tMPO2[end] *= ITensor(rho0, ri2)

    ϵ_op = ITransverse.ChainModels.epsilon_brick_ising(tp)
    ϵ_op[1] *= delta(siteind(ϵ_op,1), li1)
    ϵ_op[2] *= delta(siteind(ϵ_op,2), li2)

    LO = applyn(tMPO1, ll)
    OR = applyn(tMPO2, rr) # todo swap indices for non-symmetric MPOs

    insert!(LO.data, 1, ϵ_op[1])
    insert!(OR.data, 1, ϵ_op[2])

    ev_LOOR = overlap_noconj(LO, OR)

    deleteat!(LO.data,1)
    deleteat!(OR.data,1)

    LO[1] *= ITensor(ComplexF64[1,0,0,1], li1)
    OR[1] *= ITensor(ComplexF64[1,0,0,1], li2)

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end



""" TODO CHECK """
function expval_en_density_old(ll::MPS, rr::MPS, tp::tmpo_params)

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites_L)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites_L)

    rho0 = (tp.init_state) * (tp.init_state')
    tMPO1[end] *= ITensor(rho0, ri1)
    tMPO2[end] *= ITensor(rho0, ri2)

    temp_s = siteinds("S=1/2",2)
    os = OpSum()
    os += tp.mp.JXX, "X",1,"X",2
    os += tp.mp.hz/2,  "I",1,"Z",2
    os += tp.mp.hz/2,  "Z",1,"I",2
    os += tp.mp.λx/2,  "I",1,"X",2
    os += tp.mp.λx/2,  "X",1,"I",2

    #ϵ_op = ITensor(os, temp_s, temp_s')
    ϵ_op = MPO(os, temp_s)
    cs1 = combiner(temp_s[1], temp_s[1]')
    cs2 = combiner(temp_s[2], temp_s[2]')
    ϵ_op[1] *= cs1 
    ϵ_op[2] *= cs2 
    ϵ_op[1] *= delta(combinedind(cs1), li1)
    ϵ_op[2] *= delta(combinedind(cs2), li2)

    TTop = applyn(tMPO2, tMPO1)
    TTop[1] *= delta(li2',li2)

    tMPO_eps = deepcopy(TTop)

    tMPO_eps[1] *= ϵ_op[1]
    tMPO_eps[1] *= ϵ_op[2]

    LOO = applyn(tMPO_eps, ll)
    ev_LOOR = overlap_noconj(LOO, rr)


    tMPO_11 = TTop 
    tMPO_11[1] *= ITensor(ComplexF64[1,0,0,1], li1)
    tMPO_11[1] *= ITensor(ComplexF64[1,0,0,1], li2)

    #normalization 
    L11 = applyn(tMPO_11, ll)
    ev_L11R = overlap_noconj(L11, rr)

    return ev_LOOR/ev_L11R

end


function compute_expvals(ll::AbstractMPS, rr::AbstractMPS, op_list::Vector{String}, tp::tmpo_params)

       # ! TODO To save time, split calculation L1R and L11R in separate function called only once - make also optional ..

    if op_list[1] == "all"
        op_list = ["X", "Z", "XX", "ZZ", "eps"]
    end

    allevs = Dict()

    for op in op_list
        if op == "X"
            #println("X")
            allevs["X"] = expval_LR(ll, rr, ComplexF64[0,1,1,0], tp)
        elseif op == "Z"
            #println("Z")
            allevs["Z"] = expval_LR(ll, rr, ComplexF64[1,0,0,-1], tp)
        elseif op == "eps"
            #println("eps")
            allevs["eps"] = expval_en_density(ll, rr, tp)
        else
            @warn "$(op) not implemented"
        end
    end

    return allevs
end
