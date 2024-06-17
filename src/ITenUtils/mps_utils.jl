""" Quick random MPS for playing around """
function myrMPS(len::Int=40, chi::Int=50)
    s = siteinds("S=1/2", len)
    p = random_mps(ComplexF64, s, linkdims=chi)

    return p
end

""" Computes the overlap (ll,rr) between two MPS *without* conjugating either one
"""
function overlap_noconj(ll::MPS, rr::MPS, approx_real::Bool=false)
    siteinds(ll) != siteinds(rr) ? rr = replace_siteinds(rr, siteinds(ll)) : nothing
    overlap = inner(dag(ll),rr) 
    if approx_real && imag(overlap) < 1e-15
        return real(overlap)
    end
    
    return overlap
    
end

function ITensors.sim(psi::MPS)
    phi = deepcopy(psi)
    sim!(phi)
    return phi
end    

function sim!(psi::MPS)
    replace_siteinds!(psi, sim(siteinds(psi)))
    ll = linkinds(psi)
    sl = sim(ll)

    replaceind!(psi[1], ll[1], sl[1])
    for ii in eachindex(psi)[2:end-1]
        replaceind!(psi[ii], ll[ii-1], sl[ii-1])
        replaceind!(psi[ii], ll[ii], sl[ii])
    end
    replaceind!(psi[end], ll[end], sl[end])

end


""" Returns an MPS which is a copy of the 2nd argument
with physical indices matching those of the first one
TODO CHECK IS THIS THE SAME AS REPLACE_SITEINDS? """
function match_mps_indices(mps1::MPS, mps2::MPS)
    @assert length(mps1) == length(mps2)
    mps3 = copy(mps2)
    for (ii, Bi) in enumerate(mps2)
        mps3[ii] = Bi * delta(siteind(mps2,ii), siteind(mps1,ii))
    end

    return mps3
end

""" Changes inplace the phys indices of the 2nd argument
matching those of the first one"""
function match_mps_indices!(mps1::MPS, mps2::MPS)
    @assert length(mps1) == length(mps2)
    for ii in eachindex(mps2)
        mps2[ii] = mps2[ii] * delta(siteind(mps2,ii), siteind(mps1,ii))
    end
end

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


""" Returns a copy of psi normalized by a `factor` (spread out over all MPS tensors) """
function normbyfactor(psi::AbstractMPS, factor::Number )

    psic = deepcopy(psi)
    
    lf = log(factor)
    z = exp(lf/length(psi))

    for n in eachindex(psic)
      psic[n] ./= z
    end

    return psic

end
