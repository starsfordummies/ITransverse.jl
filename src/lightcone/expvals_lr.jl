""" Build exp value <L|O|R> for a single operator `op`.
 In order to avoid normalization issues, we build both <L|OR> and <L|1R> overlaps, 
 the exp value is given by <L|OR>/<L|1R> """
function expval_LR(ll::MPS, rr::MPS, op::Vector{<:Number}, b::FoldtMPOBlocks)

    time_sites = siteinds(rr)
    tmpo = swapprime(folded_tMPO(b, time_sites), 0, 1, "Site")
    psi1R = applyn(tmpo, rr)

    L1R = overlap_noconj(ll,psi1R)

    tmpo = swapprime(folded_tMPO(b, time_sites, op), 0, 1, "Site")
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
function expval_LR(ll::MPS, rr::MPS, opL::Vector{ComplexF64}, opR::Vector{ComplexF64}, b::FoldtMPOBlocks)

    time_sites = siteinds(ll)
    tmpo = folded_tMPO(b, time_sites, opL)
    psi_L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(folded_tMPO(b, time_sites, opR), 0, 1, "Site")
    psi_R = applyn(tmpo, rr)

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    time_sites = siteinds(ll)
    tmpo = folded_tMPO(b, time_sites)
    psi_L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(folded_tMPO(b, time_sites), 0, 1, "Site")
    psi_R = applyn(tmpo, rr)

    ev_L11R = overlap_noconj(psi_L,psi_R)

    return ev_LOOR/ev_L11R

end


""" Alternative way of computing expval_LR using open tMPOs and closing them. 
We pass the list of local operators to compute as a (regular) MPO, 
which we contract to the top of the tMPO 
It may be more flexible """
function expval_LR_open(ll::MPS, rr::MPS, ops::MPO, b::FoldtMPOBlocks)

    # TODO allow for longer MPOs (longer lists of local ops)

    @assert length(ops) == 2

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    tMPO1= folded_open_tMPO(b, time_sites_L)
    tMPO2= folded_open_tMPO(b, time_sites_R)

    rho0 = b.rho0
    tMPO1[1] = rho0 * delta(ind(rho0,1), linkind(tMPO1,1))
    tMPO2[1] = rho0 * delta(ind(rho0,1), linkind(tMPO2,1))
 
    e1 = ops[1]
    e2 = ops[2]
    e1 *= delta(siteind(ops,1), linkinds(tMPO1)[end])
    e2 *= delta(siteind(ops,2), linkinds(tMPO2)[end])

    pushfirst!(ll, ITensor(1))
    push!(ll, ITensor(1))

    LO = applyn(tMPO1, ll)
    OR = applyn(tMPO2, rr) # todo swap indices for non-symmetric MPOs

    insert!(LO.data, 1, ϵ_op[1])
    insert!(OR.data, 1, ϵ_op[2])

    ev_LOOR = overlap_noconj(LO, OR)

    deleteat!(LO.data,1)
    deleteat!(OR.data,1)

    LO[1] *= ITensor(ComplexF64[1,0,0,1], linkinds(tMPO1)[end])
    OR[1] *= ITensor(ComplexF64[1,0,0,1], linkinds(tMPO2)[end])

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end

function expval_LR_ops(ll::MPS, rr::MPS, ops::MPO, b::FoldtMPOBlocks)

    # TODO allow for longer MPOs (longer lists of local ops)

    @assert length(ops) == 2

    time_sites_L = siteinds(ll)
    new_timesite = Index(dim(time_sites_L[end]))
    push!(time_sites_L, new_timesite)
    time_sites_R = siteinds(rr)
    push!(time_sites_R, new_timesite)

    tMPO1= folded_tMPO_L(b, time_sites_L)
    tMPO2= folded_tMPO_R(b, time_sites_R)

    e1 = ops[1] * delta(siteind(ops,1), linkinds(tMPO1)[end])
    e2 = ops[2] * delta(siteind(ops,2), linkinds(tMPO2)[end])

    tMPO1[end] = e1
    tMPO2[end] = e2
    
    LO = apply_extend(tMPO1, ll)
    OR = apply_extend(tMPO2, rr) # todo swap indices for non-symmetric MPOs

    ev_LOOR = overlap_noconj(LO, OR)

    LO[end-1] *= ITensor([1,0,0,1], linkinds(LO)[end])
    OR[end-1] *= ITensor([1,0,0,1], linkinds(OR)[end])

    pop!(LO.data)
    pop!(OR.data)

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end



""" Compute exp value for energy density in Ising """
function expval_en_density(ll::MPS, rr::MPS, b::FoldtMPOBlocks)

    tp = b.tp

    time_sites_L = siteinds(ll)
    time_sites_R = siteinds(rr)

    @show siteinds(ll)
    @show siteinds(rr)
    tMPO1 = folded_open_tMPO(b, time_sites_L)
    tMPO2 = folded_open_tMPO(b, time_sites_R)

    tMPO1[1] = b.rho0 * delta(ind(b.rho0,1), linkind(tMPO1,1))
    tMPO2[1] = b.rho0 * delta(ind(b.rho0,1), linkind(tMPO2,1))

    ϵ_op = ITransverse.ChainModels.epsilon_brick_ising(tp)
    ϵ_op[1] *= delta(siteind(ϵ_op,1), linkinds(tMPO1)[end])
    ϵ_op[2] *= delta(siteind(ϵ_op,2), linkinds(tMPO2)[end])

    LO = apply_extend(tMPO1, ll)
    OR = apply_extend(tMPO2, rr) # todo swap indices for non-symmetric MPOs

    LO[end] = ϵ_op[1]
    OR[end] = ϵ_op[2]

    ev_LOOR = overlap_noconj(LO, OR)

    deleteat!(LO.data,1)
    deleteat!(OR.data,1)

    LO[end] *= ITensor(ComplexF64[1,0,0,1], linkinds(tMPO1)[end])
    OR[end] *= ITensor(ComplexF64[1,0,0,1], linkinds(tMPO2)[end])

    #normalization 
    ev_L11R = overlap_noconj(LO, OR)

    return ev_LOOR/ev_L11R

end


""" Give as input Left and Right MPS, a list of operators to build and the FoldMPOBlocks.
Returns a Dictionary with expectation values <L|O|R>/<L|1|R> """
function compute_expvals(ll::AbstractMPS, rr::AbstractMPS, op_list::Vector{String}, b::FoldtMPOBlocks)

       # ! TODO To save time, split calculation L1R and L11R in separate function called only once - make also optional ..

    if op_list[1] == "all"
        op_list = ["X", "Z", "XX", "ZZ", "eps"]
    end

    allevs = Dict()

    for op in op_list
        if op == "X"
            #println("X")
            allevs["X"] = expval_LR(ll, rr, [0,1,1,0], b)
        elseif op == "Z"
            #println("Z")
            allevs["Z"] = expval_LR(ll, rr, [1,0,0,-1], b)
        elseif op == "eps"
            #println("eps")
            ϵ_op = ITransverse.ChainModels.epsilon_brick_ising(b.tp)
            allevs["eps"] = expval_LR_ops(ll, rr, ϵ_op, b)
            #allevs["eps"] = expval_en_density(ll, rr, b)
        else
            @warn "$(op) not implemented"
        end
    end

    return allevs
end
