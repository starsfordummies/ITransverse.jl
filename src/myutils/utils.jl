
""" Quick random MPS for playing around """
function myrMPS(len::Int=40, chi::Int=50)
    s = siteinds("S=1/2", len)
    p = randomMPS(ComplexF64, s, linkdims=chi)

    return p
end

""" Computes the overlap (ll,rr) between two MPS *without* conjugating either one
"""
function overlap_noconj(ll::MPS, rr::MPS, approx_real::Bool=false)
    siteinds(ll) != siteinds(rr) ? rr = replace_siteinds(rr, siteinds(ll)) : nothing
    overlap = inner(dag(ll),rr) :: Union{Float64,ComplexF64}
    if approx_real && imag(overlap) < 1e-15
        return real(overlap)
    end
    
    return overlap
    
end



""" Returns an MPS which is a copy of the 2nd argument
with physical indices matching those of the first one"""
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


"""checks whether an MPO tensor is symmetric """
function check_symmetry_itensor_mpo(T::ITensor)
    (space_p1, space_p) = inds(T, "Site")
    (wL, wR) = inds(T, "Link")
    check_symmetry_itensor_mpo(T, wL, wR, space_p1, space_p)
end

"""checks whether an MPO tensor is symmetric, specifying the indices we want to check on"""
function check_symmetry_itensor_mpo(T::ITensor, wL::Index, wR::Index, space_p1::Index, space_p::Index)

    # check symmetry: p<->p' , wL <-> wR 
    ddelta = norm(permute(T, (wL, space_p1, space_p, wR)).tensor - permute(T, (wL, space_p, space_p1, wR)).tensor)
    if  ddelta < 1e-12
        #alternatively:
        #permute(Wc, (wL, space_p', space_p, wR)).tensor == permute(Wc, (wL, space_p, space_p', wR)).tensor
        @info("Tensor Symmetric p <->p*")
    else
        @warn("Tensor *not* symmetric p<->p*,  normdiff=$ddelta")
    end

    ddelta = norm(permute(T, (wR, space_p, space_p1, wL)).tensor - permute(T, (wL, space_p, space_p1, wR)).tensor)
    if  ddelta < 1e-12
        #permute(Wc, (wR, space_p, space_p', wL)).tensor == permute(Wc, (wL, space_p, space_p', wR)).tensor
        @info("Tensor Symmetric wL <->wR")
    else
        @warn("Tensor *not* symmetric wL<->wR, normdiff=$ddelta")
    end

end

"""checks whether a given ITensor is symmetric """
function check_symmetry_itensor_old(T::ITensor, inds_to_permute, other_indices)
  
    @assert length(inds_to_permute) == 2  # I only know how to swap pairs 

    ind_list = [inds_to_permute[1], inds_to_permute[2]]
    append!(ind_list, other_indices)

    swap_ind_list = [inds_to_permute[2], inds_to_permute[1]]
    append!(swap_ind_list, other_indices)

    ddelta = norm(permute(T, ind_list).tensor - permute(T, swap_ind_list).tensor)
    if  ddelta < 1e-12
        @info("Tensor Symmetric in $inds_to_permute")
    else
        @warn("Tensor *not* symmetric normdiff=$ddelta in $inds_to_permute")
    end

end

""" Easier to follow check for whether a given ITensor is symmetric in the `inds_to_permute` index pair 
Does a few allocations so it's maybe more expensive but it's meant for small tensors anyway """
function check_symmetry_itensor(T::ITensor, inds_to_permute)
  
    @assert length(inds_to_permute) == 2  # I only know how to swap pairs 

    other_inds = uniqueinds(T, inds_to_permute)

    Tten = Array(T, inds_to_permute, other_inds)
    Tten_swapped = Array(T, reverse(inds_to_permute), other_inds)


    ddelta = norm(Tten - Tten_swapped)
    if  ddelta < 1e-12
        @info("Tensor Symmetric in $inds_to_permute")
    else
        @warn("Tensor *not* symmetric normdiff=$ddelta in $inds_to_permute")
    end

end









function check_diag_matrix(d::Matrix, cutoff::Float64=1e-6)
    delta_diag = norm(d - Diagonal(d))/norm(d)
    if delta_diag > cutoff
        println("Warning, matrix non diagonal: $delta_diag")
        return false
    end
    return true
end


function check_id_matrix(d::Matrix, cutoff::Float64=1e-6)
    if size(d,1) == size(d,2)
        delta_diag = norm(d - I(size(d,1)))/norm(d)
        if delta_diag > cutoff
            println("Not identity: off by(norm) $delta_diag")
            return false
        end
        return true
    else
        println("Not even square? $(size(d))")
        return false
    end
end

function isid(a::ITensor, cutoff::Float64=1e-8)
    @assert ndims(a) == 2
    @assert size(a,1) == size(a,2)

    am = array(a)

    check_id_matrix(am)
end

function plot_matrix(a::Matrix)
    heatmap(1:size(a,1),
           1:size(a,2), abs.(a),
           c=cgrad([:blue, :white,:red, :yellow]),
           xlabel="i", ylabel="j",
           title="matrix")
end

function plot_matrix(a::ITensor)
    @assert order(a) == 2
    plot_matrix(matrix(a))
end
