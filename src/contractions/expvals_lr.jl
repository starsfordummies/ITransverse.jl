"""
    _time_sites(psi, b)

The site Indices of `psi` that are *time* steps, so that an operator column rebuilt over
them lines up with `psi`.

A non-product bottom boundary (`b.rho0`) is appended as its own site and is afterwards
indistinguishable from a time site, so the count is read off the blocks the caller hands in:
`n_boundary_sites(b.rho0)`. That is right exactly when `psi` was built with the same `rho0`,
which is the caller's responsibility - `folded_tMPS` lets you override it.

The *top* needs no count: on this path a vector's top is always the product
`vectorized_identity`, because operators are inserted by the expval helpers (which close
each rebuilt column with `fold_op`) rather than baked into the vector at build time. A
vector carrying a non-product top belongs to a hand-contracted network instead - see
[`_check_product_top`](@ref), which catches most such vectors on the way in.
"""
function _time_sites(psi::TMPSorMPS, b)
    ss = siteinds(unsided(psi))
    return ss[(1 + n_boundary_sites(b.rho0)):end]
end

"""
    _check_product_top(psi, b, what)

Refuse a vector whose top looks like a non-product boundary rather than a time site.

The expval helpers close each rebuilt column with `fold_op` - the operator - so there is no
room for a boundary tensor on top as well; contracting anyway silently uses the wrong
network. A boundary tensor appended at the top has the same *rank* as an end-of-chain site
tensor, so the only structural handle is its dimension: every time site carries the folded
physical dimension `dim(b.iP)`, a boundary bond carries the boundary MPS' bond dimension.

!!! warning "Not a complete check"
    A non-product top whose bond dimension happens to equal the folded physical dimension is
    indistinguishable here and slips through. Build such networks by hand (a column with the
    bulk, rank-3 boundary, paired with `apply_column` / `overlap_noconj`) rather than through
    these helpers.
"""
function _check_product_top(psi::TMPSorMPS, b, what::AbstractString)
    ts = _time_sites(psi, b)
    isempty(ts) && return nothing
    d = dim(b.iP)
    dim(last(ts)) == d || error("""
        the top site of $(what) has dimension $(dim(last(ts))), not the folded physical
        dimension $(d): it looks like a non-product boundary state rather than a time site.
        The expectation-value helpers close each rebuilt column with `fold_op`, so they
        cannot also carry that boundary - contract such a network explicitly instead, with
        `folded_tMPO(b, ts; fold_op=<rank-3 boundary>)` and `apply_column` /
        `overlap_noconj`.""")
    return nothing
end


""" Given <L|, MPO,|R> computes exp value <L|op_mpo|R>  (here L is *not* conjugated!)
in a supposedly efficient way. No normalization and no compression is done here.  """
function expval_LR(ll::TMPSorMPS, op_mpo::MPO, rr::TMPSorMPS; match_inds::Bool=false)
    ll, rr = unsided(ll), unsided(rr)  # accept a tagged boundary vector, work on the MPS

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
function expval_LR(ll::TMPSorMPS, opL::MPO, opR::MPO, rr::TMPSorMPS; match_inds::Bool=false)
    ll, rr = unsided(ll), unsided(rr)  # accept a tagged boundary vector, work on the MPS

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


function expval_LR(ll::TMPSorMPS, rr::TMPSorMPS, operators::Tuple, b::FoldtMPOBlocks; match_inds::Bool=false)
    expval_LR(ll, rr,  operators..., b; match_inds)
end



""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately. """
function expval_LR(ll::TMPSorMPS, rr::TMPSorMPS, op::AbstractVector, b::FoldtMPOBlocks; match_inds::Bool=false)

    # Assuming here siteinds(ll) and (rr) match
    _check_product_top(ll, b, "the left vector")
    _check_product_top(rr, b, "the right vector")
    time_sites = _time_sites(rr, b)
    tmpo = folded_tMPO(b, time_sites; fold_op=op)
    expval_LR(ll, tmpo, rr; match_inds)
    
end


""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` """ 
function expval_LR(ll::TMPSorMPS, rr::TMPSorMPS, opL::AbstractVector, opR::AbstractVector,
                   b::FoldtMPOBlocks; match_inds::Bool=false)

    _check_product_top(ll, b, "the left vector")
    _check_product_top(rr, b, "the right vector")
    time_sites = _time_sites(ll, b)

    tmpoL = folded_tMPO(b, time_sites, fold_op=opL)

    time_sites = _time_sites(rr, b)
    tmpoR = folded_tMPO(b, time_sites, fold_op=opR)

    expval_LR(unsided(ll), tmpoL, tmpoR, unsided(rr); match_inds)

end





"""
    expval_LR_ops(ll, rr, ops::MPS, b::FoldtMPOBlocks)

⟨L|O|R⟩ for an operator `O` spread over `length(ops)` adjacent sites, given as a *folded*
operator MPS: each site tensor carries the vectorized physical leg (tagged `"Site"`,
dimension `dim(b.iP)`) and the spatial bonds to its neighbours.

An MPS is the right container: folding has already vectorized the operator's two physical
legs into one, so what is left is a chain of dim-`d²` legs joined by spatial bonds. One
column is built per site with that tensor capping its top ([`folded_tMPO_op`](@ref)), and
the `k` columns are contracted in a single staggered sweep - column `j` primed by `k-j` so
the site legs chain, and the caps primed only on their own internal link so the spatial
bonds they share stay at plev 0 and contract pairwise.

Not normalized: divide by the same expression with identity caps.
"""
function expval_LR_ops(ll::TMPSorMPS, rr::TMPSorMPS, ops::MPS, b::FoldtMPOBlocks)
    k = length(ops)
    k >= 1 || throw(ArgumentError("need at least one operator site, got $(k)"))
    _check_product_top(ll, b, "the left vector")
    _check_product_top(rr, b, "the right vector")

    L, R = unsided(ll), unsided(rr)
    ops = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, L), ops)

    cols = [folded_tMPO_op(b, _time_sites(rr, b), ops[j]) for j in 1:k]

    O = ITensors.OneITensor()
    for ii in eachindex(L)
        O = O * R[ii]
        for j in k:-1:1
            O = O * prime(cols[j][ii], k - j)
        end
        O = O * prime(L[ii], k)
    end
    # the caps: prime only the link into each column, so the shared spatial bonds contract
    for j in k:-1:1
        lnk = commonind(cols[j][end], cols[j][end - 1])
        O = O * prime(cols[j][end], k - j, lnk)
    end
    return scalar(O)
end

""" A folded operator handed over as an `MPO` container: its site tensors have already been
vectorized (one combined leg per site), so re-tag them and use the `MPS` method. Kept
because `epsilon_brick_ising` used to return this shape. """
function expval_LR_ops(ll::TMPSorMPS, rr::TMPSorMPS, ops::MPO, b::FoldtMPOBlocks)
    return expval_LR_ops(ll, rr, _as_folded_operator(ops), b)
end

""" Re-type a vectorized operator stored in an `MPO` container as the `MPS` it structurally
is, tagging each combined physical leg `"Site"`. """
function _as_folded_operator(ops::MPO)
    return MPS([settags(ops[j], "Site", siteind(ops, j)) for j in eachindex(ops)])
end



""" Give as input Left and Right MPS, a list of operators to build and the FoldMPOBlocks.
Returns a Dictionary with expectation values <L|O|R>/<L|1|R> """
function compute_expvals(ll::TMPSorMPS, rr::TMPSorMPS, op_list, b::FoldtMPOBlocks)

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
function expval_LR_apply(ll::TMPSorMPS, op_mpo::MPO, rr::TMPSorMPS)
    ll, rr = unsided(ll), unsided(rr)  # accept a tagged boundary vector, work on the MPS

    @assert length(ll) == length(op_mpo) == length(rr)
    orr = applyn(op_mpo, rr)

    ev_LOR = overlap_noconj(ll,orr)

    return ev_LOR

end


""" Build exp value <L|O|R> for a single vectorized operator `op`, given as a 1D array 
   Does *NOT* normalize here by <L|1|R>, need to do it separately.
   Slower version which uses ITensors' apply(), allows to truncate intermediate MPO """
function expval_LR_apply(ll::TMPSorMPS, rr::TMPSorMPS, op::AbstractVector, b::FoldtMPOBlocks; maxdim=nothing)

    _check_product_top(ll, b, "the left vector")
    _check_product_top(rr, b, "the right vector")
    time_sites = _time_sites(rr, b)
    tmpo = folded_tMPO(b, time_sites; fold_op=op)
    psiOR = isnothing(maxdim) ? applyn(tmpo, unsided(rr)) : apply(tmpo, unsided(rr); alg="naive", maxdim)
    LOR = overlap_noconj(unsided(ll), psiOR)

    return LOR

end

""" Build exp value <L|opLopR|R> for a pair of local operator `opL` and `opR` using apply() """ 
function expval_LR_apply(ll::TMPSorMPS, rr::TMPSorMPS, opL::AbstractVector, opR::AbstractVector, b::FoldtMPOBlocks)

    _check_product_top(ll, b, "the left vector")
    _check_product_top(rr, b, "the right vector")
    time_sites = _time_sites(ll, b)
    tmpo = folded_tMPO(b, time_sites, fold_op=opL)
    psi_L = applyn(tmpo, unsided(ll))

    time_sites = _time_sites(rr, b)
    tmpo = swapprime(folded_tMPO(b, time_sites, fold_op=opR), 0, 1, "Site")
    psi_R = applyn(tmpo, unsided(rr))

    ev_LOOR = overlap_noconj(psi_L,psi_R)

    return ev_LOOR

end