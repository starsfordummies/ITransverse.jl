""" A boundary vector, with or without the [`SidedMPS`](@ref) metadata. """
const TMPSLike = Union{AbstractMPS, SidedMPS}

""" The plain MPS behind a boundary vector. """
_psi(psi::AbstractMPS) = psi
_psi(s::SidedMPS) = MPS(s)

""" Time sites of a transverse MPS built with the blocks `b`: drops the extra site(s) that a
non-product boundary state adds at *either* end of the chain (see `boundary_tensor`), so
that a tMPO rebuilt from them matches the input MPS.

A `SidedMPS` carries both counts and is used as-is. A plain `MPS` records nothing, so the
bottom count is inferred from `b.rho0` - correct only if the vector was built with that same
`rho0` - and the top is assumed to be product. Build the vector with `sided=true` when
either end is non-product: a boundary site appended at the top is indistinguishable from a
time site here, and rebuilding a tMPO over it contracts the wrong network. """
function _time_sites(psi::TMPSLike, b)
    nbot = psi isa SidedMPS ? n_boundary_bottom(psi) : n_boundary_sites(b.rho0)
    ntop = n_boundary_top(psi)
    ss = siteinds(_psi(psi))
    return ss[(1 + nbot):(end - ntop)]
end

""" The expval helpers rebuild the operator column over the *time* sites and close its top
with `fold_op` - which is the operator. A vector whose own top is a non-product boundary
therefore has one site the rebuilt column cannot match: the column would have to carry that
boundary as well (its bulk, rank-3 form) *and* the operator. That is a network the helpers
do not build, so refuse it rather than contract the wrong one.

Only reachable when the metadata says so, i.e. for vectors built with `sided=true`; a plain
`MPS` cannot report a top boundary at all (see [`n_boundary_top`](@ref)). """
function _check_no_top_boundary(psi::TMPSLike, what::AbstractString)
    ntop = n_boundary_top(psi)
    ntop == 0 || error("""
        $(what) carries $(ntop) non-product boundary site(s) at the top of the chain, which
        the expectation-value helpers do not support: the operator column they rebuild
        closes its own top with `fold_op`, so it cannot also carry that boundary.
        Contract such a network explicitly instead - build the column with the bulk
        boundary, `folded_tMPO(b, ts; fold_op=<rank-3 boundary>)`, and pair it with
        `applyn` / `overlap_noconj`.""")
    return nothing
end


""" Given <L|, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
in a supposedly efficient way. No normalization and no compression is done here.  """
function expval_LR(ll::MPS, op_mpo::MPO, rr::MPS; match_inds::Bool=false)

    if match_inds
        op_mpo = replace_siteinds(op_mpo, siteinds(rr), siteinds(ll)')
    end

    nL = length(ll)
    nR = length(rr)


  
    #O = ll[1]' * (op_mpo[1] * rr[1])
    O = ITensors.OneITensor()

    for ii in 1:min(nL, nR)
        O = O * rr[ii]
        O = O * op_mpo[ii]
        O = O * ll[ii]'
    end

    if nL > nR 
        for ii = nR+1:nL
            O = O * op_mpo[ii]
            O = O * ll[ii]'
        end
    elseif nL < nR
        for ii = nL+1:nR
            O = O * rr[ii]
            O = O * op_mpo[ii]
        end
    end

    return scalar(O)

end


""" Given <L|MPO, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
in a supposedly efficient way. No normalization nor compression is done here.  """
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


function expval_LR(ll::TMPSLike, rr::TMPSLike, operators::Tuple, b::FoldtMPOBlocks; match_inds::Bool=false)
    expval_LR(ll, rr,  operators..., b; match_inds)
end



""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately. """
function expval_LR(ll::TMPSLike, rr::TMPSLike, op::AbstractVector, b::FoldtMPOBlocks; match_inds::Bool=false)

    # Assuming here siteinds(ll) and (rr) match
    _check_no_top_boundary(ll, "the left vector")
    _check_no_top_boundary(rr, "the right vector")
    time_sites = _time_sites(rr, b)
    tmpo = folded_tMPO(b, time_sites; fold_op=op)
    expval_LR(ll, tmpo, rr; match_inds)
    
end


""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` """ 
function expval_LR(ll::TMPSLike, rr::TMPSLike, opL::AbstractVector, opR::AbstractVector, b::FoldtMPOBlocks)

    _check_no_top_boundary(ll, "the left vector")
    _check_no_top_boundary(rr, "the right vector")
    time_sites = _time_sites(ll, b)
    # TODO CHECK do we need to swap legs on the left ? 
    #tmpoL = swapprime(folded_tMPO(b, time_sites, opL), 0, 1, "Site")
    tmpoL = folded_tMPO(b, time_sites, fold_op=opL)

    time_sites = _time_sites(rr, b)
    tmpoR = folded_tMPO(b, time_sites, fold_op=opR)

    expval_LR(_psi(ll), tmpoL, tmpoR, _psi(rr))

end





""" Expval of a local operator which we feed as a standard *spatial MPO*. 
We do this by building tMPO with one extra site on top, and replace it by the relevant operator 
Warning, this does *not* compute the normalization """
function expval_LR_ops(ll::TMPSLike, rr::TMPSLike, ops::MPO, b::FoldtMPOBlocks)

    # TODO: make for MPOs with length > 2 
    @assert length(ops) == 2

    ops = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, _psi(ll)), ops)


    _check_no_top_boundary(ll, "the left vector")
    _check_no_top_boundary(rr, "the right vector")

    time_sites_L = _time_sites(ll, b)
    new_timesite = Index(dim(time_sites_L[end]))
    push!(time_sites_L, new_timesite)
    time_sites_R = _time_sites(rr, b)
    push!(time_sites_R, new_timesite)

    tMPO1= folded_tMPO_ext(b, time_sites_L, LR=:left)
    tMPO2= folded_tMPO_ext(b, time_sites_R; LR=:right)

    e1 = ops[1] * delta(siteind(ops,1), linkinds(tMPO1)[end])
    e2 = ops[2] * delta(siteind(ops,2), linkinds(tMPO2)[end])

    tMPO1[end] = e1
    tMPO2[end] = e2
    
    LO = applyns(tMPO1, _psi(ll))
    OR = applyn(tMPO2, _psi(rr))

    ev_LOOR = overlap_noconj(LO, OR)

    return ev_LOOR

end



""" Give as input Left and Right MPS, a list of operators to build and the FoldMPOBlocks.
Returns a Dictionary with expectation values <L|O|R>/<L|1|R> """
function compute_expvals(ll::TMPSLike, rr::TMPSLike, op_list, b::FoldtMPOBlocks)

    # TODO truncate on apply MPO in expval_... 

    if op_list == "all"
        op_list = ["X", "Z", "Pz", "Sp", "Sm", "XX", "ZZ", "eps_ising"]
    end

    allevs = Dict{String,ComplexF64}()

    #Normalization 
    idN = vectorized_identity(dim(b.iR))
    ev_L1R = expval_LR(ll, rr, idN, b)

    #two-col exp value is expensive, only compute if necessary
    ev_L11R = ("XX" in op_list) || ("ZZ" in op_list) || ("eps_ising" in op_list) ? expval_LR(ll, rr, (idN, idN), b) : 1.0

    for op in op_list
        if op == "eps_ising"  # do this separately
            ϵ_op = ITransverse.epsilon_brick_ising(b.tp.mp)
            allevs[op] = expval_LR_ops(ll, rr, ϵ_op, b)/ev_L11R
        elseif op == "XX" 
            allevs[op] = expval_LR(ll, rr, [0,1,1,0], [0,1,1,0], b)/ev_L11R
        elseif  op == "ZZ"
            allevs[op] = expval_LR(ll, rr, [1,0,0,-1], [1,0,0,-1], b)/ev_L11R
        elseif  op == "Pz"
            allevs[op] = expval_LR(ll, rr, [1,0,0,0], b)/ev_L1R
  
        else  # Basically all one-site operators should be handled by ITensors (+ appropriate overloading)

            opv = vectorized_op(op, b.tp.mp.phys_site)
            allevs[op] = expval_LR(ll, rr, opv, b)/ev_L1R
        end
    end

    return allevs
end





""" Given <L|, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
Version with ITensors' apply(), in principle slower 
No normalization is done here.  """
function expval_LR_apply(ll::MPS, op_mpo::MPO, rr::MPS)

    @assert length(ll) == length(op_mpo) == length(rr)
    orr = applyn(op_mpo, rr)

    ev_LOR = overlap_noconj(ll,orr)

    return ev_LOR

end


""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately.
   Slower version which uses ITensors' apply(), allows to truncate intermediate MPO """
function expval_LR_apply(ll::TMPSLike, rr::TMPSLike, op::AbstractVector, b::FoldtMPOBlocks; maxdim=nothing)

    _check_no_top_boundary(ll, "the left vector")
    _check_no_top_boundary(rr, "the right vector")
    time_sites = _time_sites(rr, b)
    tmpo = folded_tMPO(b, time_sites; fold_op=op)
    psiOR = isnothing(maxdim) ? applyn(tmpo, _psi(rr)) : apply(tmpo, _psi(rr); alg="naive", maxdim)
    LOR = overlap_noconj(_psi(ll), psiOR)

    return LOR

end

""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` using apply() """ 
function expval_LR_apply(ll::TMPSLike, rr::TMPSLike, opL::AbstractVector, opR::AbstractVector, b::FoldtMPOBlocks)

    _check_no_top_boundary(ll, "the left vector")
    _check_no_top_boundary(rr, "the right vector")
    time_sites = _time_sites(ll, b)
    tmpo = folded_tMPO(b, time_sites, fold_op=opL)
    psi_L = applyn(tmpo, _psi(ll))

    time_sites = _time_sites(rr, b)
    tmpo = swapprime(folded_tMPO(b, time_sites, fold_op=opR), 0, 1, "Site")
    psi_R = applyn(tmpo, _psi(rr))

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    return ev_LOOR

end