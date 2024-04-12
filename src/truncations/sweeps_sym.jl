
#
# * Symmetric sweeps - could be eventually superseded by iGensors functions 
# 

""" 
Symmetric case: Truncates a single MPS optimizing overlap (L|L) (no conj)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns L_ortho, ents_sites
"""
# function truncate_normalize_sweep_sym(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100)

#     mpslen = length(left_mps)

#     L_ortho = orthogonalize(left_mps,1, normalize=false)

#     XUinv= ITensor(1.)

#     ents_sites = Vector{Float64}()

#     for ii =1:mpslen-1

#         Ai = XUinv * L_ortho[ii]
#         vR = linkind(L_ortho,ii)
    
#         left_env = Ai * prime(Ai, vR)

#         @assert order(left_env) == 2

#         # legacy 
#         #U,S = symmetric_svd_arr(left_env, svd_cutoff=svd_cutoff, chi_max=chi_max)

#         U,S = symm_svd(left_env, vR, cutoff=svd_cutoff, maxdim=chi_max)

#         # orig version (works)
#         sqS = sqrt.(S)
#         isqS = sqS.^(-1)

#         XU = dag(U) * isqS
#         #XU = transpose(conj(U)) * isqS
#         XUinv = sqS * U
#         #

#         #=new version 

#         sqS = sqrt.(diag(S))
#         isqS = sqS.^(-1)
        
#         XU = dag(U) * diagITensor(isqS.storage.data, inds(S))
#         XUinv = diagITensor(sqS.storage.data, inds(S)) * U
#         =#

#         #L_ortho[ii] =  Ai * XU  #replacetags(Ai * XU , "Link,v", "Link,v=$ii")
#         L_ortho[ii] =  replacetags(Ai, "v", "Link,v=$(ii-1)") * replacetags(XU, "v", "Link,v=$ii") 
#         push!(ents_sites, log(sum(S)))
#     end

#     # the last two 
#     An = XUinv * L_ortho[mpslen]

#     overlap = An * An 

    
#     @assert order(overlap) == 0 
#     # normalize overlap to 1 at each step 
#     L_ortho[mpslen] =  replacetags(An, "v", "Link,v=$(mpslen-1)") /sqrt(overlap[1])


#     return L_ortho, ents_sites

# end



""" 
Symmetric case: Truncates a single MPS optimizing overlap (L|L) (no conj)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns L_ortho, ents_sites
"""
function truncate_normalize_sweep_sym!(left_mps::MPS; svd_cutoff::Float64, chi_max::Int, method::String)

    mpslen = length(left_mps)

    orthogonalize!(left_mps,1) # orthogonalize! should NOT normalize MPS 

    #CHECK: does this help ?
    #normalize_gen!(left_mps)

    @debug norm_gen(left_mps)

    XUinv= ITensor(1.)
    left_env = ITensor(1.)
    Ai = ITensor(1.)

    ents_sites = [] # Vector{Float64}()

    for ii = 1:mpslen-1

        Ai = XUinv * left_mps[ii]

        left_env = left_env * Ai
        left_env = left_env * Ai'
        left_env *= delta(siteind(left_mps,ii),siteind(left_mps,ii)')

        if method == "SVDold"
            U,S = symmetric_svd_arr(left_env, svd_cutoff=svd_cutoff, chi_max=chi_max)

            sqS = sqrt.(diag(S))
            isqS = sqS.^(-1)
            
            XU = dag(U) * diagITensor(isqS.storage.data, inds(S))
            XUinv = diagITensor(sqS.storage.data, inds(S)) * U

        elseif method == "SVD"
            F = symm_svd(left_env, ind(left_env,1), cutoff=svd_cutoff, maxdim=chi_max)
            U = F.U
            S = F.S

            sqS = S.^(0.5)
            isqS = sqS.^(-1)
            
            #@assert inds(S) ==  inds(isqS)
 
            #XU = dag(U) * diagITensor(isqS.storage.data, inds(S))
            XU = dag(U) * isqS
            XUinv = sqS * U

            # test = dag(U) * U' * delta(F.u, F.u')

            # isid(XU * prime(XUinv,"CMB"))

        elseif method == "EIG"
            F = symm_oeig(left_env, ind(left_env,1), cutoff=svd_cutoff)
            #@show dump(F)
            U = F.V
            S = F.D

            sqS = S.^(0.5)
            isqS = sqS.^(-1)

            XU = U * isqS
            XUinv = sqS * U

            #@show inds(XU)
            #@show inds(left_env)

            #@show isid(XU * XUinv)

        end

        #slightly faster version (check)

        #

        left_mps[ii] =  Ai * XU  

        # TODO NORMALIZE HERE ?? 
        #left_mps[ii] = left_mps[ii] / norm_gen(left_mps[ii])

        left_env =  left_mps[ii] * prime(left_mps[ii], inds(left_mps[ii])[end])

        #@show S
        #@show sum(S)
        push!(ents_sites, log(sum(S)))
    end

    # the last two 
    An = XUinv * left_mps[mpslen]

    overlap = An * An 

    if abs(scalar(overlap)) > 1e20 || abs(scalar(overlap)) < 1e-20
        @warn ("Careful! overlap overflowing? = $scalar(overlap)")
    end

    # normalize overlap to 1 on last matrix 
    left_mps[mpslen] =  An /sqrt(scalar(overlap))


    @debug "Sweep done, normalization $(overlap_noconj(left_mps, left_mps))"
    #sleep(1)

    # Fix indices here ? 

    #@show linkinds(left_mps)

    # At the end, better relabeling of indices 
    for (ii,li) in enumerate(linkinds(left_mps))
        newlink = Index(dim(li), "Link,l=$ii")
        left_mps[ii] *= delta(li, newlink)
        left_mps[ii+1] *= delta(li, newlink)
        #@show inds(left_mps[ii])
    end

    # for ii in eachindex(left_mps)
    #     replacetags!(left_mps[ii], "v" => "Link,v=$ii")
    # end

    #@show linkinds(left_mps)

    return ents_sites

end



function truncate_normalize_sweep_sym_ite!(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100, method="gen_one")
    orthogonalize!(left_mps,1)
    _, _, ents = orthogonalize_gen_ents!(left_mps, length(left_mps); cutoff=svd_cutoff, normalize=true, method)
    return ents
end




""" 
Symmetric case sweep to bring to gen. RIGHT canonical form
TODO havent' checked it works yet 
"""
function truncate_normalize_sweep_sym_right(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100)

    mpslen = length(left_mps)

    # bring to LEFT standard canonical form 
    L_ortho = orthogonalize(left_mps,mpslen, normalize=false)

    XUinv= ITensor(1.)

    ents_sites = Vector{Float64}()

    for ii =mpslen:-1:2
        Ai = XUinv * L_ortho[ii]

        vL = linkind(L_ortho,ii-1)
    
        # assume it's identities to the right (ie. we're carrying along gen. canon form)
        right_env = Ai * prime(Ai, vL)

        @assert order(right_env) == 2

        U,S = symmetric_svd_arr(right_env, svd_cutoff=svd_cutoff, chi_max=chi_max)

        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        XUinv = sqS * U

        L_ortho[ii] =  replacetags(Ai, "v", "Link,v=$(ii)") * replacetags(XU, "v", "Link,v=$(ii-1)") 

        push!(ents_sites, log(sum(S)))
    end

    # the last two 
    An = XUinv * L_ortho[1]

    overlap = An * An 

    
    @assert order(overlap) == 0 
    # normalize overlap to 1 at each step 
    L_ortho[1] =  replacetags(An, "v", "Link,v=1") /sqrt(overlap[1])


    return L_ortho, ents_sites

end




""" Check that an MPS is in (generalized-symmetric) left canonical form,
ie. that when we contract everything from the left to the right we get identities
Should be superseded by check_gen_ortho() in iGensors """
function _check_gencan_left_sym(in_mps::MPS, verbose::Bool=false)
    if verbose
        println("Checking LEFT gen/sym form")
    end

    if abs(overlap_noconj(in_mps,in_mps) - 1.) > 1e-7 || abs(1. - scalar(in_mps[end]*in_mps[end])) > 1e-7
        println("overlap(LEFTgen) = $(overlap_noconj(in_mps,in_mps))), alt = $(scalar(in_mps[end]*in_mps[end]))")
    end

    mpslen = length(in_mps)
    # Start from the operator/final state side (for me that's on the left)
    left_env = ITensor(1.)
    for (ii, Ai) in enumerate(in_mps[1:end-1])
        left_env =  left_env * Ai
        left_env = left_env * prime(Ai, commoninds(Ai,linkinds(in_mps)))   #* delta(wLa, wLb) )
        @assert order(left_env) == 2
        delta_norm = norm(array(left_env) - I(size(left_env)[1])) 

        if norm(array(left_env)- diagm(diag(array(left_env)))) > 0.1
            println("[L]non-can@[$ii]")
        end

        if delta_norm > 0.01
            println("[L]non-can@[$ii], $delta_norm")
        end
    end
    if verbose
        println("Done checking LEFT gen/sym form")
    end

end


""" Check that an MPS is in (generalized-symmetric) *right* canonical form
Should be superseded by check_gen_ortho() in my iGensors """
function _check_gencan_right_sym(in_mps::MPS, verbose::Bool=false)
    if verbose
        println("Checking RIGHT gen/sym form")
    end
    
    if abs(overlap_noconj(in_mps,in_mps) - 1.) > 1e-7 || abs(1. - scalar(in_mps[1]*in_mps[1])) > 1e-7
        println("overlap = $(overlap_noconj(in_mps,in_mps))), alt = $(scalar(in_mps[1]*in_mps[1]))")
    end

    mpslen = length(in_mps)
    # Start from the operator/final state side (for me that's on the left)
    right_env = ITensor(1.)
    for (ii, Ai) in enumerate(in_mps[end:-1:2])
        right_env =  right_env * Ai
        right_env = right_env * prime(Ai, commoninds(Ai,linkinds(in_mps)))    #* delta(wLa, wLb) )
        @assert order(right_env) == 2
        if norm(array(right_env)- diagm(diag(array(right_env)))) > 0.1
            println("[R]non-diag@[$ii]")
        end
        delta_norm = norm(array(right_env) - I(size(right_env)[1])) 
        if delta_norm > 0.01
            println("[R]non-can@[$ii], $delta_norm")
        end
    end
    if verbose
        println("Done checking RIGHT gen/sym form")
    end
    
end




""" Check that an MPS is in (generalized-symmetric) mixed canonical form at site b 
Should be superseded by check_gen_ortho() in my iGensors """
function _check_gencan_mixed_sym(in_mps::MPS, b::Int64, half::Int64, verbose::Bool=false)


    if half == 1 # Left sweep
        bleft = b
        bright = b+2 
    else
        bleft = b-1
        bright = b+1 
    end

    mpslen = length(in_mps)

    if b == 1 || b == mpslen
        return
    end

    if verbose
        println("Checking MIXED gen/sym form")
    end
    
    # Start from the operator/final state side (for me that's on the left)
    right_env = ITensor(1.)
    for ii = mpslen:-1:bright
        Ai = in_mps[ii]
        right_env =  right_env * Ai
        right_env = right_env * prime(Ai, commoninds(Ai,linkinds(in_mps)))    #* delta(wLa, wLb) )
        @assert order(right_env) == 2
        if norm(array(right_env)- Diagonal(array(right_env))) > 0.1
            println("[<R]non-diag@[$ii]")
        end
        delta_norm = norm(array(right_env) - I(size(right_env)[1]))
        if delta_norm > 0.1
            println("[<R]quite non-can@[$ii], $delta_norm")
        end
    end

    left_env = ITensor(1.)
    for ii =1:bleft
        Ai = in_mps[ii]
        left_env =  left_env * Ai
        left_env = left_env * prime(Ai, commoninds(Ai,linkinds(in_mps)))   #* delta(wLa, wLb) )
        @assert order(left_env) == 2
        delta_norm = norm(array(left_env) - I(size(left_env)[1])) /norm(array(left_env))

        if norm(array(left_env)- Diagonal(array(left_env))) > 0.1
            println("[L>]non-diag@[$ii]")
        end

        if delta_norm > 0.1
            println("[L>]quite non-can@[$ii], $delta_norm")
        end
    end
    if verbose
        println("Done checking LEFT gen/sym form")
    end

    if verbose
        println("Done checking RIGHT gen/sym form")
    end

    if abs(overlap_noconj(in_mps,in_mps) - 1) > 1e-5
        println("overlap[mix] = $(overlap_noconj(in_mps,in_mps))")
    end
    
end

