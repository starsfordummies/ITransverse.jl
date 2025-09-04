""" 
 Environment structure for transverse contraction. 
    For a finite network of N columns, we build separately N-1 (left and right) environments

 L1 is the left edge vector (first column), RN-1 is the right one (last column)

 That is, the starting are L_1 = <L|,  R_{N-1} = |R>

    Eg. for N=6 we have

    L1(-L2(-L3(-L4(-L5(
        T2  T3  T4  T5  
       )R1-)R2-)R3-)R4-)R5

     So Li+1 = Li * Ti+1  ;  Ri = Ti+1 * Ri+1

  The full contraction of the network is given by Li * Ri, for all i 
"""
struct Environments{T}
    envs::Vector{T}
    norms::Vector{Float64}
end

""" The constructor fills the first (or last, depending on `LR`) element with the input MPS 
and normalizes it (saving its norm in envs.norms) """
function Environments(n::Int, boundary::MPS, LR::String) 

    envs = Vector{MPS}(undef, n-1) 
    norms = fill(1.0, n-1) 

    boundary = orthogonalize(boundary, length(boundary))
    norm_boundary = norm(boundary)
    boundary = normalize(boundary)

    if LR == "L"
        envs[1] = boundary
        norms[1] = norm_boundary
    elseif LR == "R"
        envs[end] = boundary 
        norms[end] = norm_boundary
    else
        error("specify whether left or right env")
    end

    return Environments(envs, norms)
end

Base.size(a::Environments) = length(a.envs)
Base.eachindex(env::Environments) = eachindex(env.envs)  # Forward to the underlying Vector
Base.lastindex(env::Environments) = length(env.envs)
# Optionally, implement size if you want to use size-based functions
Base.length(env::Environments) = length(env.envs)

Base.getindex(a::Environments, i::Int) = a.envs[i]

function Base.setindex!(a::Environments, v, i::Int)
    a.envs[i] = v
end

Base.iterate(env::Environments, state=1) = state <= length(env.envs) ? (env.envs[state], state + 1) : nothing

function Base.pop!(env::Environments) 
     pop!(env.envs)
     pop!(env.norms)
end
function Base.popfirst!(env::Environments)
     popfirst!(env.envs)
     popfirst!(env.norms)
end

function ITensorMPS.maxlinkdim(a::Environments)
    return maximum(maxlinkdim.(a.envs))::Int
end


function overlap_at(left_envs::Environments, right_envs::Environments, jj::Int)
    # for sanity checks..
    NN = length(left_envs)
    overlap = overlap_noconj(left_envs[jj], right_envs[jj])
    #push!(overlaps_nofactors, overlap)
    for kk = 1:jj 
        overlap *= left_envs.norms[kk]
    end
    for kk = jj:NN
        overlap *= right_envs.norms[kk] 
    end

    return overlap

end

function overlap_mid(left_envs::Environments, right_envs::Environments)
    overlap_at(left_envs, right_envs, div(length(left_envs),2))
end

""" Compute overlaps between envs at all sites and their (absolute) stdev """
function overlap_envs(left_envs::Environments, right_envs::Environments; verbose::Bool=false)
  
    NN = length(left_envs.envs)
    overlaps = ComplexF64[]
    for jj in 1:NN-1
        overlap = overlap_noconj(left_envs[jj], right_envs[jj])
        for kk = 1:jj
            overlap *= left_envs.norms[kk]
        end
        for kk = jj:NN
            overlap *= right_envs.norms[kk]
        end

        push!(overlaps, overlap)
    end

    if verbose
        @show overlaps
    end
    return mean(overlaps) ::ComplexF64, std(overlaps) ::Float64

end


""" Compute mean Generalized SVD entropy for the input environment sets """
function mean_gen_vn_ents(left_envs::Environments, right_envs::Environments)
    NN = length(left_envs.envs)
    eents = []
    for jj in 2:NN-1
        SSVDGen = generalized_svd_vn_entropy(left_envs[jj], right_envs[jj])
        push!(eents, SSVDGen)
    end

    return mean(eents), std(eents) 

end

""" Compute mean Generalized Renyi2  """
function mean_gen_renyi2_ents(left_envs::Environments, right_envs::Environments)
    NN = length(left_envs.envs)
    eents = []
    for jj in 2:NN-1
        S2Gen = gen_renyi2(left_envs[jj], right_envs[jj])
        push!(eents, S2Gen)
    end

    return mean(eents), std(eents) 

end