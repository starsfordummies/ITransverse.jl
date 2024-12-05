""" Quick random MPS for playing around """
function quick_mps(len::Int=40, chi::Int=50)
    s = siteinds("S=1/2", len)
    p = random_mps(ComplexF64, s, linkdims=chi)

    return p
end

""" Computes the overlap (ll,rr) between two MPS *without* conjugating either one.
If siteinds(ll) and siteinds(rr) do not match, it matches them before contracting.
"""
function overlap_noconj_ite(ll::MPS, rr::MPS, approx_real::Bool=false)
    siteinds(ll) != siteinds(rr) ? rr = replace_siteinds(rr, siteinds(ll)) : nothing
    overlap = inner(dag(ll),rr) 
    if approx_real && imag(overlap) < 1e-15
        return real(overlap)
    end
    
    return overlap
    
end

""" Computes the overlap (ll,rr) between two MPS *without* conjugating either one.
The "generalized norm" of an MPS should be sqrt(overlap_noconj(psi,psi)).
"""
function overlap_noconj(ll::MPS, rr::MPS, approx_real::Bool=false)
    #siteinds(ll) != siteinds(rr) ? rr = replace_siteinds(rr, siteinds(ll)) : nothing

    if !ITensorMPS.hassameinds(siteinds, ll, rr)
        @warn "L and R don't have the same physical indices, correcting "
        rr = replace_siteinds(rr, siteinds(ll))
    end

    ll = sim(linkinds, ll)

    overlap = ll[1] * rr[1]
    for ii in eachindex(ll)[2:end]
        overlap = (overlap * rr[ii]) * ll[ii]
    end

    if approx_real && imag(overlap) < 1e-14
        return real(overlap)
    end
    
    return scalar(overlap)
    
end


""" Given `mps1` and `mps2`, returns a copy of `mps2`
with physical indices matching those of the first one
"""
function match_siteinds(mps1::MPS, mps2::MPS)
    replace_siteinds(mps2, siteinds(mps1))
end

""" Given `mps1` and `mps2`, replaces `mps2` siteinds with those of `mps1`
"""
function match_siteinds!(mps1::MPS, mps2::MPS)
    replace_siteinds!(mps2, siteinds(mps1))
end

""" Given two MPOs, replaces the siteinds of the *second* with those of the first """
function match_siteinds!(mpo1::MPO, mpo2::MPO)
    for j in eachindex(mpo1)
        sold = siteind(mpo2,j)
        snew = siteinds(mpo1,j)
        replaceinds!(mpo2[j], sold' => snew')
        replaceinds!(mpo2[j], sold => snew)
    end
end

function replace_linkinds!(psi::MPS, newtags::String="")
    li = linkinds(psi)
    newli = [removetags(l, tags(l)) for l in li]
    newli2 = [addtags(newli[ii], newtags*"$ii") for ii in eachindex(newli)]
    
    for ii in eachindex(psi)
        replaceinds!(psi[ii], li, newli2)
    end

    return newli2
end


""" TODO have a look at this if we can do better """
function extend_mps(in_mps::MPS, new_sites)

    new_mps = deepcopy(in_mps)

    # add trivial link to the last matrix of in_mps
    last_mat = new_mps[end]
    lastinds = inds(last_mat)
    last_r = Index(1, "Link,l="*string(length(in_mps)))
    new_mps[end] = ITensor(last_mat.tensor.storage.data, (lastinds,last_r))


    # add one matrix at the end 
    new_last = ITensor(2., last_r, new_sites[end])

    push!(new_mps.data, new_last)

    return new_mps

end

function extend_sites(old_sites::Vector{<:Index}, sitetype::String, sitetags::String)
    new_sites = copy(old_sites)
    final_site = addtags(siteind(sitetype, length(old_sites)+1), sitetags)

    push!(new_sites, final_site)
    return new_sites
end


function extend_mps_factorize(in_mps::MPS; site_type::String="S=1/2", tags::String="")

    new_sites = extend_sites(siteinds(in_mps), site_type, tags)
    lastinds = inds(in_mps[end])

    last_mat = in_mps[end] * ITensor(1., new_sites[end])

    a, b =  factorize(last_mat, lastinds, tags="v", cutoff=1e-14)

    new_mps = deepcopy(in_mps)

    new_mps[end] = a
    push!(new_mps.data, b)
    
    return new_mps

end




function extend_mps_factorize(in_mps::MPS, new_sites::Vector{<:Index}) 

    @assert length(new_sites) == length(in_mps) + 1 

    lastinds = inds(in_mps[end])

    last_mat = in_mps[end] * ITensor(1., new_sites[end])

    # u,s,vd = svd(last_mat, lastinds, lefttags="Link,l=$(length(in_mps))", righttags="Link,l=$(length(in_mps))", cutoff=1e-14)
    # u_sqs = u * sqrt.(s) 
    # sqs_vd = sqrt.(s) * vd 

    #a, b =  factorize(last_mat, lastinds, tags="Link,l=$(length(in_mps))", cutoff=1e-14)
    a, b =  factorize(last_mat, lastinds, tags="v", cutoff=1e-14)


    new_mps = deepcopy(in_mps)

    # for ii in eachindex(new_mps)
    #     replace_siteinds!(new_mps[ii], new_sites[ii])
    # end
    # for ii in eachindex(in_mps)
    #     replaceind!(in_mps[ii], siteinds(in_mps)[ii], new_sites[ii])
    # end

    new_mps[end] = a
    push!(new_mps.data, b)
    
    replace_siteinds!(new_mps, new_sites)

    return new_mps

end

function extend_mps_v(in_mps::MPS, new_sites::Vector{<:Index})

    new_mps = deepcopy(in_mps)

    # add trivial link to the last matrix of in_mps
    last_mat = new_mps[end]
    lastinds = inds(last_mat)
    last_l = linkinds(in_mps)[end]
    last_r = Index(1, "v")
    println(last_l, new_sites[end-1], last_r)
    new_mps[end] = ITensor(last_mat.tensor.storage.data, (last_l, new_sites[end-1],last_r))


    # add one matrix at the end 
    new_last = ITensor(1., last_r, new_sites[end])

    push!(new_mps.data, new_last)

    # just to be sure.. (inplace?)
    replace_siteinds(new_mps,new_sites)

    @assert siteinds(new_mps) == new_sites
    return new_mps

end


""" Returns a copy of psi normalized by a `factor` (spread out over all MPS tensors using log) """
function normbyfactor(psi::AbstractMPS, factor::Number )

    psic = deepcopy(psi)
    
    lf = log(factor)
    z = exp(lf/length(psi))

    for n in eachindex(psic)
      psic[n] ./= z
    end

    # When we do this, we are breaking canonical forms so reset llims
    ITensorMPS.setleftlim!(psic,1)
    ITensorMPS.setrightlim!(psic,length(psic))

    return psic

end

""" Shorthand for simple apply with naive algorithm and no truncation """
function applyn(O::MPO, psi::AbstractMPS)
    apply(O, psi, alg="naive", truncate=false)
end

""" Shorthand for apply with no truncation + swap indices """
function applyns(O::MPO, psi::AbstractMPS)
    apply(swapprime(O, 0, 1, "Site"), psi, alg="naive", truncate=false)
end

function ITensorMPS.replace_siteinds!(M::MPO, sites)
    for j in eachindex(M)
      sj = siteind(M, j)
      M[j] = replaceinds(M[j], (sj,sj') => (sites[j],sites[j]'))
    end
    return M
  end


""" Returns an MPS with a gauge fixed to a *left* form such that 
1) all tensors M[2:end] are in left canonical form 
2) the first tensor contracts with its conj to a diagonal matrix 

"""
function gaugefix_left(psi::MPS)
    psi_work = orthogonalize(psi,length(psi))
    orthogonalize!(psi_work,1)

    lenv_1 = psi_work[1] * prime(dag(psi_work[1]), linkind(psi_work,1))
    vals, vecs = eigen(lenv_1)
    psi_work[1] = psi_work[1] * vecs
    psi_work[2] = dag(vecs) * psi_work[2] 

    return psi_work
end


function myrMPO(sites::Vector{<:Index}; linkdims::Int)
    o = random_mpo(sites)
    for ii = 2:linkdims
        o += random_mpo(sites)
    end
    return o
  end
