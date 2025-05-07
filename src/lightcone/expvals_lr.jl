
""" Given <L|, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
Version with ITensors' apply(), in principle slower 
No normalization is done here.  """
function expval_LR_apply(ll::MPS, op_mpo::MPO, rr::MPS)

    @assert length(ll) == length(op_mpo) == length(rr)
    orr = applyn(op_mpo, rr)

    ev_LOR = overlap_noconj(ll,orr)

    return ev_LOR

end

""" Given <L|, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
in a supposedly efficient way. No normalization is done here.  """
function expval_LR(ll::MPS, op_mpo::MPO, rr::MPS; match_inds::Bool=false)

    if match_inds
        if siteinds(ll) != siteinds(rr)
             rr = replace_siteinds(rr, siteinds(ll)) 
        end
    end

    @assert length(ll) == length(op_mpo) == length(rr)
  
    #O = ll[1]' * (op_mpo[1] * rr[1])
    O = ITensor(1)

    for ii in eachindex(ll)#[2:end]
        O = O * rr[ii]
        O = O * op_mpo[ii]
        O = O * ll[ii]'
    end

    return scalar(O)

end


""" Given <L|MPO, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
in a supposedly efficient way. No normalization is done here.  """
function expval_LR(ll::MPS, opL::MPO, opR::MPO, rr::MPS; match_inds::Bool=false)

    if match_inds
        if siteinds(ll) != siteinds(rr)
             rr = replace_siteinds(rr, siteinds(ll)) 
        end
    end

    @assert length(ll) == length(opL) == length(opR) == length(rr)
  
    O = ll[1]'' * (opL[1]' * (opR[1] * rr[1]))

    for ii in eachindex(ll)[2:end]
        O = O * rr[ii]
        O = O * opR[ii]
        O = O * opL[ii]'
        O = O * ll[ii]''
    end

    return scalar(O)

end





""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately.
   Slower version which uses ITensors' apply(), allows to truncate intermediate MPO """
function expval_LR_apply(ll::MPS, rr::MPS, op::AbstractVector, b::FoldtMPOBlocks; maxdim=nothing)

    time_sites = siteinds(rr)
    tmpo = folded_tMPO(b, time_sites, op)
    psiOR = isnothing(maxdim) ? applyn(tmpo, rr) : apply(tmpo,rr; alg="naive", maxdim)
    LOR = overlap_noconj(ll,psiOR)

    return LOR

end

""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately. """
function expval_LR(ll::MPS, rr::MPS, op::AbstractVector, b::FoldtMPOBlocks)

    # Assuming here siteinds(ll) and (rr) match
    time_sites = siteinds(rr)
    tmpo = folded_tMPO(b, time_sites; fold_op=op)
    expval_LR(ll, tmpo, rr)
    
end


""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` """ 
function expval_LR(ll::MPS, rr::MPS, opL::AbstractVector, opR::AbstractVector, b::FoldtMPOBlocks)

    time_sites = siteinds(ll)
    # TODO CHECK do we need to swap legs on the left ? 
    #tmpoL = swapprime(folded_tMPO(b, time_sites, opL), 0, 1, "Site")
    tmpoL = folded_tMPO(b, time_sites, opL)

    time_sites = siteinds(rr)
    tmpoR = folded_tMPO(b, time_sites, opR)

    expval_LR(ll, opL, opR, rr)

end


""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` using apply() """ 
function expval_LR_apply(ll::MPS, rr::MPS, opL::AbstractVector, opR::AbstractVector, b::FoldtMPOBlocks)

    time_sites = siteinds(ll)
    tmpo = folded_tMPO(b, time_sites, opL)
    psi_L = applyn(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(folded_tMPO(b, time_sites, opR), 0, 1, "Site")
    psi_R = applyn(tmpo, rr)

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    # time_sites = siteinds(ll)
    # tmpo = folded_tMPO(b, time_sites)
    # psi_L = applyn(tmpo, ll)

    # time_sites = siteinds(rr)
    # tmpo = swapprime(folded_tMPO(b, time_sites), 0, 1, "Site")
    # psi_R = applyn(tmpo, rr)

    # ev_L11R = overlap_noconj(psi_L,psi_R)

    return ev_LOOR

end



""" Expval of a list of local operators, which we feed as a standard *spatial MPO*. 
We do this by building tMPO with one extra site on top, and replace it by the relevant operator 
Warning, this does *not* compute the normalization """
function expval_LR_ops(ll::MPS, rr::MPS, ops::MPO, b::FoldtMPOBlocks)

    # TODO: make for MPOs with length > 2 
    @assert length(ops) == 2

    ops = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, ll), ops)


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

    return ev_LOOR

end



""" Give as input Left and Right MPS, a list of operators to build and the FoldMPOBlocks.
Returns a Dictionary with expectation values <L|O|R>/<L|1|R> """
function compute_expvals(ll::AbstractMPS, rr::AbstractMPS, op_list::Vector{String}, b::FoldtMPOBlocks)

    # TODO truncate on apply MPO in expval_... 

    if op_list[1] == "all"
        op_list = ["X", "Z", "Pz", "Sp", "Sm", "XX", "ZZ", "eps"]
    end

    s_folded = inds(b.WWc)[end]

    phys_space = string(split(string(tags(s_folded)),",")[1])[2:end]
    s_unfolded = siteinds(phys_space,1)[1]
    local_dim = dim(s_unfolded)
    Id = [diagm(ones(local_dim))...]
    
    allevs = Dict{String,ComplexF64}()


    ev_L1R = expval_LR(ll, rr, Id, b)

    #two-col exp value is expensive, only compute if necessary
    ev_L11R = haskey(op_list, "XX") || haskey(op_list, "ZZ") || haskey(op_list, "eps") ? expval_LR(ll, rr, [1,0,0,1], [1,0,0,1], b) : 1.0

    for op_String in op_list
        
        if op_String == "Pz"
            allevs[op_String] = expval_LR(ll, rr, [1,zeros(local_dim^2-1)...], b)/ev_L1R
        # elseif op == "Sp"
        #         allevs[op] = expval_LR(ll, rr, [0,1,0,0], b)/ev_L1R
        # elseif op == "Sm"
        #         allevs[op] = expval_LR(ll, rr, [0,0,1,0], b)/ev_L1R
        elseif op_String == "XX"
                σx = [dim(s) * matrix(op_String(s,"Sx"))...]
                allevs[op_String] = expval_LR(ll, rr, σx, σx, b)/ev_L1R
        elseif op_String == "ZZ"
                σz = [dim(s) * matrix(op_String(s,"Sz"))...]
                allevs[op_String] = expval_LR(ll, rr, σz, σz, b)/ev_L1R
        elseif op_String == "eps"
            ϵ_op = ITransverse.ChainModels.epsilon_brick_ising(b.tp.mp)
            allevs[op_String] = expval_LR_ops(ll, rr, ϵ_op, b)/ev_L11R
        else
            current_mat_op = [matrix(op(s_unfolded,op_String))...]
            allevs[op_String] = expval_LR(ll, rr, current_mat_op, b)/ev_L1R
        end
    end

    return allevs
end






""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately.
   Slower version which uses ITensors' apply(), allows to truncate intermediate MPO """
function expval_LR_apply_list(ll::MPS, rr::MPS, op_list::AbstractVector, b::FoldtMPOBlocks,  b_im::FoldtMPOBlocks; maxdim=256)

    time_sites = siteinds(rr)

    psiOR = deepcopy(rr)
    for op in op_list
        tmpo = folded_tMPO(b, b_im, time_sites, op)
        psiOR = isnothing(maxdim) ? applyn(tmpo, psiOR) : apply(tmpo,psiOR; alg="naive", maxdim)
    end

    LOR = overlap_noconj(ll,psiOR)


    return LOR

end



""" Given an input MPS |B>, an operator list, and a "middle" operator X, builds tMPO columns with all the operators in the list 
and applies them to the MPS, building O1*O2*..*ON|B> = |OB> . Then computes the exp value <BO|X|OB>, so it
should be thought as a symmetric string +1 extra operator in the middle."""
function expval_LR_apply_list_sym(rr::MPS, op_list::AbstractVector, op_mid::String, b::FoldtMPOBlocks,  b_im::FoldtMPOBlocks; maxdim=256, cutoff=1e-10, method="RDM")

    time_sites = siteinds(rr)
    tmpo_id  = folded_tMPO_doublebeta(b, b_im, time_sites)

    psiOR = deepcopy(rr)
 
    for (ii, str_op) in enumerate(op_list)

        if str_op == "Id"
            op = ComplexF64[1,0,0,1]
        elseif str_op == "Pz"
            op = ComplexF64[1,0,0,0]
        else
            @error "No valid operator given"
        end

        tmpo = folded_tMPO_doublebeta(b, b_im, time_sites, op)

        if method == "RTM"
            #@info "1: " linkdims(psiIR)


            #TODO CHECK Do we want to correct by overlap before or after truncation?
            #overlap_IR = overlap_noconj(psiIR,psiIR)
            #@info ii, " applying ", str_op
            #@info "Before apply:" ITransverse.overlap_noconj(psiOR, psiOR)
            psiOR = applyn(tmpo, psiOR) 
            #@info "After apply:" ITransverse.overlap_noconj(psiOR, psiOR)
            psiOR, _, overlap_OR = truncate_rsweep_sym(psiOR; cutoff, chi_max=maxdim, method="SVD")
            #@info "After trunc:" ITransverse.overlap_noconj(psiOR, psiOR)
            #psiOR = ITransverse.normbyfactor(psiOR, sqrt(overlap_OR))
            #@info "After normalization:" ITransverse.overlap_noconj(psiOR, psiOR)
        else # RDM
            psiOR = isnothing(maxdim) ? applyn(tmpo, psiOR) : apply(tmpo,psiOR; alg="naive", maxdim, cutoff)
        end
        # Try to normalize along the way ? 
        #psiOR = psiOR / norm1

        #next!(p; showvalues = [(:Info,"chi=$(maxlinkdim(psiIR))|chi=$(maxlinkdim(psiOR))" )])

    end

    LR = overlap_noconj(psiOR,psiOR)

    if op_mid == "Id"
        tmpo = folded_tMPO_doublebeta(b, b_im, time_sites)
    elseif op_mid == "Pz"
        tmpo = folded_tMPO_doublebeta(b, b_im, time_sites, ComplexF64[1,0,0,0])
    else 
        @error "Wrong op ?", op_mid
    end

    OOR = applyn(tmpo, psiOR)
    LOR = overlap_noconj(psiOR, OOR) 

    return LOR, LR

end



""" Given an input MPS |B>, an operator list, and a "middle" operator X, builds tMPO columns with all the operators in the list 
and applies them to the MPS, building O1*O2*..*ON|B> = |OB> . Then computes the exp value <BO|X|OB>, so it
should be thought as a symmetric string +1 extra operator in the middle."""
function expval_LR_apply_list_sym_2(rr::MPS, op_list::AbstractVector, b::FoldtMPOBlocks,  b_im::FoldtMPOBlocks; maxdim=256, cutoff=1e-10, method="RDM")

    time_sites = siteinds(rr)
    psiOR = deepcopy(rr)

    norm_factors = []
 
    for str_op in op_list

        if str_op == "Id"
            op = ComplexF64[1,0,0,1]
        elseif str_op == "Pz"
            op = ComplexF64[1,0,0,0]
        else
            @error "Operator $(str_op) not implemented yet"
        end

        tmpo = folded_tMPO_doublebeta(b, b_im, time_sites, op)

        if method == "RTM"
          
            psiOR = applyn(tmpo, psiOR) 
            psiOR, _, overlap_OR = truncate_rsweep_sym(psiOR; cutoff, chi_max=maxdim, method="SVD")
            push!(norm_factors, overlap_OR)
            psiOR = ITransverse.normbyfactor(psiOR, sqrt(overlap_OR))
        else # RDM
            psiOR = isnothing(maxdim) ? applyn(tmpo, psiOR) : apply(tmpo,psiOR; alg="naive", maxdim, cutoff)
            @info maxlinkdim(psiOR), norm(psiOR)
            push!(norm_factors, 1.)
        end
    
    end

    LR = overlap_noconj(psiOR,psiOR)

    return LR * prod(norm_factors), LR, norm_factors, psiOR

end