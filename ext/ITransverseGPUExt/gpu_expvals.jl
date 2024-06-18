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



function ITransverse.gpu_expval_LL_sym(ll::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

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

function gpu_expval_en_density(ll::MPS, rr::MPS, tp::tmpo_params)

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    tMPO1, li1, ri1 = build_folded_open_tMPO(tp, time_sites_L)
    tMPO2, li2, ri2 = build_folded_open_tMPO(tp, time_sites_R)

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
    ϵ_op = NDTensors.cu(MPO(os, temp_s))
    cs1 = combiner(temp_s[1], temp_s[1]')
    cs2 = combiner(temp_s[2], temp_s[2]')
    ϵ_op[1] *= cs1 
    ϵ_op[2] *= cs2 
    ϵ_op[1] *= delta(combinedind(cs1), li1)
    ϵ_op[2] *= delta(combinedind(cs2), li2)

    LO = applys(NDTensors.cu(tMPO1), ll)
    OR = applys(NDTensors.cu(tMPO2), rr) # todo swap indices for non-symmetric MPOs

    insert!(LO.data, 1, ϵ_op[1])
    insert!(OR.data, 1, ϵ_op[2])

    ev_LOOR = overlap_noconj(LO, OR)

    deleteat!(LO.data,1)
    deleteat!(OR.data,1)

    LO[1] *= NDTensors.cu(ITensor(ComplexF64[1,0,0,1], li1))
    OR[1] *= NDTensors.cu(ITensor(ComplexF64[1,0,0,1], li2))

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end


function gpu_compute_expvals(ll::AbstractMPS, rr::AbstractMPS, op_list::Vector{String}, tp::tmpo_params)

   # ! TODO To save time, split calculation L1R and L11R in separate function called only once - make also optional ..

    if op_list[1] == "all"
        op_list = ["X", "Z", "XX", "ZZ", "eps"]
    end

    allevs = Dict()

    for op in op_list
        if op == "X"
            #println("X")
            allevs["X"] = gpu_expval_LR(ll, rr, ComplexF64[0,1,1,0], tp)
        elseif op == "Z"
            #println("Z")
            allevs["Z"] = gpu_expval_LR(ll, rr, ComplexF64[1,0,0,-1], tp)
        elseif op == "eps"
            #println("eps")
            allevs["eps"] = gpu_expval_en_density(ll, rr, tp)
        end
    end

    return allevs
end
