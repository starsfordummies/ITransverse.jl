import ITensorMPS: replace_siteinds!, replace_siteinds

function pMPS(ss::Vector{<:Index}, site_tensor::AbstractVector{<:Number})
    psi = MPS(ss)
    for j in eachindex(psi)
        psi[j] = ITensor(site_tensor, inds(psi[j]))
    end
    return psi
end

function pMPS(N::Int, site_tensor::AbstractVector{<:Number})
    ss = siteinds("S=1/2", N)
    pMPS(ss, site_tensor)
end

function pMPS(N::Int, site_tensor::ITensor)
    ss = siteinds("S=1/2", N)
    @assert ndims(site_tensor) == 1
    pMPS(ss, site_tensor.tensor.storage)
end


function pMPS(ss::Vector{<:Index}, site_tensors::AbstractVector{<:AbstractVector{<:Number}})
    psi = MPS(ss)
    for j in eachindex(psi)
        psi[j] = ITensor(site_tensors[j], inds(psi[j]))
    end
    return psi
end

function pMPS(site_tensors::AbstractVector{<:AbstractVector{<:Number}})
    N = length(site_tensors)
    ss = siteinds("S=1/2", N)
    pMPS(ss, site_tensors)
end


""" Quick random MPS for playing around """
function quick_mps(len::Int=10, chi::Int=32)
    s = siteinds("S=1/2", len)
    p = random_mps(ComplexF64, s, linkdims=chi)

    return p
end

function quick_mpo(len::Int=10, chi::Int=2)
    s = siteinds("S=1/2", len)
    myrMPO(s; linkdims=chi)
end

""" Return triple (psi,phi,o)  sharing the same sites"""
function quick_psipsio(len::Int=10)
    s = siteinds("S=1/2", len)
    psi = random_mps(ComplexF64, s, linkdims=12)
    phi = random_mps(ComplexF64, s, linkdims=20)

    oo = myrMPO(s; linkdims=2)

    return psi,phi, oo 
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
function overlap_noconj(ll::MPS, rr::MPS; fast::Bool=false)

    if !fast 
        return inner(conj(ll),rr) 
    else


        if !ITensorMPS.hassameinds(siteinds, ll, rr)
            @warn "L and R don't have the same physical indices, correcting "
            rr = replace_siteinds(rr, siteinds(ll))
        end

        ll = sim(linkinds, ll)

        overlap = ll[1] * rr[1]
        for ii in eachindex(ll)[2:end]
            overlap = (overlap * rr[ii]) * ll[ii]
        end
        
        return scalar(overlap) #:: ComplexF64
    end
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


function extend_sites(old_sites::Vector{<:Index}, sitetype::String, sitetags::String)
    new_sites = copy(old_sites)
    final_site = addtags(siteind(sitetype, length(old_sites)+1), sitetags)

    push!(new_sites, final_site)
    return new_sites
end


""" Returns a copy of psi normalized by a `factor` (spread out over all MPS tensors using log) """
function normbyfactor(psi::AbstractMPS, factor::Number )

    #elt = promote_type(eltype(factor), promote_itensor_eltype(psi)) # promote_itensor_eltype(psi)
    #psic = adapt(elt, psi)

    psic = deepcopy(psi)
    lf = log(factor)
    z = exp(lf/length(psi))

    for n in eachindex(psic)
      psic[n] ./= z
    end

    # When we do this, we are breaking canonical forms so reset llims
    ITensorMPS.reset_ortho_lims!(psic)

    return psic

end

""" Given two MPS, normalizes one of them to make their overlap to 1 and their norms as close to 1 as possible """
function normalize_for_overlap!(psi::AbstractMPS, phi::AbstractMPS)

    orthogonalize!(psi,1)
    orthogonalize!(phi,1)

    overlap = overlap_noconj(psi,phi)
    abs_overlap = abs(overlap)

    if abs_overlap < 1 
        # Overlap is small → normalize the one with smaller norm
        if norm(psi) < norm(phi)
            psi[1] = psi[1]/overlap
        else
            phi[1] = phi[1]/overlap
        end
    else
        # Overlap is large → normalize the one with larger norm
        if norm(psi) < norm(phi)
            phi[1] = phi[1]/overlap
        else
            psi[1] = psi[1]/overlap
        end
    end

end


# function ITensorMPS.replace_siteinds!(M::MPO, sites)
#     for j in eachindex(M)
#       sj = siteind(M, j)
#       M[j] = replaceinds(M[j], (sj,sj') => (sites[j],sites[j]'))
#     end
#     return M
#   end

function ITensorMPS.replace_siteinds(W::MPO, new_in_sites, new_out_sites=dag(new_in_sites)')
  replace_siteinds!(copy(W), new_in_sites, new_out_sites)
end

function ITensorMPS.replace_siteinds!(W::MPO, new_in_sites, new_out_sites=dag(new_in_sites)')
  @assert length(new_in_sites) == length(W)
  # Here I assume that the MPO has the "standard" physical index notation  p - p' 
  si = firstsiteinds(W)
  for ii in eachindex(W)
    replaceinds!(W[ii], si[ii] => new_in_sites[ii])
    replaceinds!(W[ii], si[ii]' => new_out_sites[ii])
  end
  return W 
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


""" Builds a random MPO with `linkdims` bond dimension """
function myrMPO(sites::Vector{<:Index}; linkdims::Int)
    if linkdims > 15  # Break up into smaller pieces and sum them afterwards
        ns = div(linkdims,10)
        o = myrMPO(sites, linkdims=linkdims % 10)
        @showprogress for _ = 1:ns
            o += myrMPO(sites, linkdims=10)
        end

    else
        o = random_mpo(sites)
        for _ = 2:linkdims
            o += random_mpo(sites)
        end
    end

    return o

end

""" Given a list of tensors in the form [vL, p, p', vR], builds an MPO.
Note that the index order sanity is left to the user!  """
function mpo_from_arrays(array_list, ss = siteinds(size(array_list[1],2), length(array_list)))
    @assert length(array_list) > 2
    @assert length(ss) == length(array_list)
    @assert size(array_list[2], 1) ==  size(array_list[2], 4) 
    @assert size(array_list[2], 2) ==  size(array_list[2], 3) 

    L = length(array_list)

    linkdim = size(array_list[2], 1)
    linkinds = [Index(linkdim, "Link,l=$(ii)") for ii = 1:L-1]

    mpo_tensors =  [ITensor() for _ in 1:L]
    mpo_tensors[1] = ITensor(array_list[1], ss[1], ss[1]', linkinds[1])
    for ii = 2:L-1
        mpo_tensors[ii] = ITensor(array_list[ii], linkinds[ii-1],  ss[ii], ss[ii]', linkinds[ii])
    end
    mpo_tensors[end] = ITensor(array_list[end], linkinds[end], ss[end], ss[end]')

    return MPO(mpo_tensors)
end


function productMPO(phys_sites, list_of_operators=fill("I",length(phys_sites)))
    Ws = [op(list_of_operators[ii], phys_sites, ii) for ii in eachindex(phys_sites)]
    MPO(Ws)
end

""" Returns a product MPS made of (folded) operators. Defaults to all identities """
function folded_productMPS(phys_sites, 
    list_of_operators=fill("I",length(phys_sites)), 
    folded_sites=siteinds(dim(phys_sites[1])^2, length(phys_sites)))
    
    NN = length(phys_sites)
    links = [Index(1, "Link, n=$n") for n in 1:NN-1]
    WsFold = Vector{ITensor}(undef, NN)
    WsFold[1] = ITensor(vectorized_op(list_of_operators[1], phys_sites[1]), folded_sites[1], links[1])
    for jj = 2:NN-1
        WsFold[jj] = ITensor(vectorized_op(list_of_operators[jj], phys_sites[jj]), links[jj-1], folded_sites[jj], links[jj])
    end
    WsFold[end] = ITensor(vectorized_op(list_of_operators[end], phys_sites[end]), links[end], folded_sites[end])
    
    return MPS(WsFold)

end


""" Removes trivial links from a product state """
function delete_link_from_prodMPS(psi)
    delete_link_from_prodMPS!(copy(psi))
end

""" Removes trivial links from a product state, inplace version """
function delete_link_from_prodMPS!(psi::AbstractMPS)
    if maxlinkdim(psi) == 1
        ss = siteinds(psi)
        for ii in eachindex(psi)
            psi[ii] = ITensor(array(psi[ii]), ss[ii])
        end
    end
    return psi
end


function fidelity(psi::MPS, phi::MPS)
    phi = copy(phi)
    match_siteinds!(psi, phi)
    return sqrt(abs2(inner(psi,phi))/norm(psi)^2/norm(phi)^2) 
end

""" Returns log10-fidelity *per site* of two MPS """
function logfidelity(psi::MPS, phi::MPS)
    phi = copy(phi)
    match_siteinds!(psi, phi)
    return (log10(abs(inner(psi,phi))) -log10(norm(psi)) - log10(norm(phi)))/length(psi)
end

""" Measures infidelity 1 - |<psi|phi>|^2/(<psi|psi><phi|phi>) """
function infidelity(psi::MPS, phi::MPS)
    return 1. - fidelity(psi,phi)
end




function check_mps_sanity(psi::MPS)
    good = if length(siteinds(psi)) != length(psi.data)
        @warn "length/sites: $(length(siteinds(psi))) != $(length(psi.data))"
        false
    elseif length(linkinds(psi)) != length(psi.data)-1
        false
    elseif ndims(psi[1]) != 2 || ndims(psi[end]) != 2 || !all(x -> ndims(x) == 3, psi[2:end-1])
        false
    else
        true
    end
    return good
end

# i1 = Index(3,"a")
# i2 = Index(4,"b")
""" Experimental predictable combiner - returns a combiner whose combinedind id is the sum of the ids of the
two indices I'm joining, hoping that nothing bad goes on - Should ensure that each time we pcombine two equal indices,
we get the same combinedind  """
function pcombiner(i1::Index, i2::Index; kwargs...)
    cc = combiner(i1, i2; kwargs...)
    ci = combinedind(cc)
    ci_alt  = Index(i1.id + i2.id, ci.space, ci.dir, ci.tags, ci.plev)
    replaceind!(cc, ci => ci_alt)
    return cc
end

# ck = combiner(i1,i2)


""" This just daggers all the inds of an MPS/MPO, no conjugation is made """
function daginds!(psi::AbstractMPS)
    for T in psi 
        ITensors.setinds!(T, dag(inds(T)))
    end
    return psi 
end

function daginds(psi::AbstractMPS)
    daginds!(copy(psi))
end

""" Returns an MPS with the (folded) MPO for a local operator `local_op` at `site_op` """
function vectorized_local_op(ss::Vector{<:Index}; local_op::String="Id", site_op=div(length(ss)+1,2))

    local_ops = [op("Id", s) for s in ss]
    local_ops[site_op] = op(local_op, ss[site_op])

    o_local_ops = MPO(local_ops)

    vo_local_ops, combiners = vectorize_mpo(o_local_ops)
end
