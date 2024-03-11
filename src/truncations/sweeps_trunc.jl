#import ITensors

"""
Truncates optimizing the overlap (L|R)
Cutoff on SV of the transition matrix, given by `cutoff` param

For this, we first bring to the usual *right* canonical form both MPS (ortho center on 1st site), \\
then we build environments L|R from the *left* and truncate on their SVDs (or EIGs depending on `method`)

So this can be seen as a "RL: Right(can)Left(gen)" sweep 
"""
function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS, truncp::trunc_params)
    truncate_normalize_sweep(left_mps, right_mps, method=truncp.method, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
end


function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    # bring to "standard" right canonical forms individually - ortho center on the 1st site 
    # making copies along the way 

    # L_ortho = orthogonalize(left_mps,  1)
    # R_ortho = orthogonalize(right_mps, 1)

    # # ! does this change anything ? doesn't seem like it 
    L_ortho = orthogonalize(left_mps,  mpslen)
    R_ortho = orthogonalize(right_mps, mpslen)
    orthogonalize!(L_ortho,1)
    orthogonalize!(R_ortho,1)
    normalize!(L_ortho)
    normalize!(R_ortho)
    

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = Vector{ComplexF64}()

    # Left sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = deltaS * Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        if method == "EIG"  # Truncation based on eigenvalues

            F = eigtrunc(left_env, cutoff, chi_max)
            # eigen(left_env, iL, iR; cutoff, maxdim=chi_max, ishermitian=false)
            U = F.V
            S = F.D
            Uinv = F.Vt 

            ind_v = commonind(S,Uinv)
            ind_u = commonind(S, U)
            link_v = uniqueind(Uinv, S)
            link_u = uniqueind(U, S)

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = (Uinv*delta(ind_v, ind_u) * delta(link_v, link_u) ) * isqS  
            XUinv = sqS * U

            XV = (U * delta( ind_v, ind_u)*delta( link_v, link_u )) * isqS # same as [p]inv(Vdag) * isqS ?
            #XV = (U * delta( inds(Vdag, "v"), inds(U, "u"))*delta( inds(Vdag, "Link"), inds(U, "Link"))) * isqS # same as [p]inv(Vdag) * isqS ?
            XVinv = sqS * Uinv

        elseif method == "SVD" 
            
            #@info "cutoff = nothing"
            #U,S,Vdag = svd(left_env, ind(left_env,1); cutoff=nothing, maxdim=chi_max)
            #@info SVs = diag(matrix(S))

            U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)
            #@info "cutoff = $cutoff"
            #@info SVs = diag(matrix(S))

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS
            XUinv = sqS * U

            XV = dag(Vdag) * isqS
            XVinv = sqS * Vdag

            #@show (XU * XUinv)
            #@show (XV * XVinv)
            

        else
            throw(ArgumentError("need to specify method EIG|SVD - current method=$method"))
        end

        deltaS = delta(inds(S))

        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #println(S)
        # compute some "effective entropy" using the SVs here
        # gen_ent_cut = 0.
        # for n=1:dim(S,1)
        #     p = S[n,n]        # I don't think we need the ^2 here 
        #     gen_ent_cut -= p * log(p)
        # end

        # Eg renyi 2 should be cheap? just for convergence - always abs() even if it's complex 
        #gen_ent_cut = sum(abs2.(S)/(norm(S)^2))
        #push!(ents_sites, gen_ent_cut)

        #! This could be nasty if we have imaginary stuff...
        #! should not be a problem for SVDs though
        push!(ents_sites, log(sum(S)))
      
    end

    # the last two 
    An = XUinv * L_ortho[end]
    Bn = XVinv * R_ortho[end]

    overlap = deltaS * ( An * Bn ) 


    
    # TODO CHECK - should I normalize always by overlap here? 
    # this could lead to large roundoff errors when eg. overlap is zero
    @assert order(overlap) == 0 
    
    # normalize overlap to 1 at each step 
    L_ortho[end] =  An/sqrt(scalar(overlap))
    R_ortho[end] =  Bn/sqrt(scalar(overlap))

    # check that we're in the appropriate gen ortho form at the end 
    #check_gencan_left_sym_phipsi(L_ortho, R_ortho)

    #println("div by overlap $(overlap)")
    #println(complex(overlap))

    return L_ortho, R_ortho, ents_sites

end

function check_equivalence_svd(left_mps::MPS, right_mps::MPS)

    mpslen = length(left_mps)
    @info "Checking if SVs are the same "
    @show mpslen

    # bring to "standard" right canonical forms individually - ortho center on the 1st site 
    # making copies along the way 

    L_ortho = orthogonalize(left_mps,  1, normalize=false)
    R_ortho = orthogonalize(right_mps, 1, normalize=false)

    # # ! does this change anything ? doesn't seem like it 
    # L_ortho = orthogonalize(left_mps,  mpslen)
    # R_ortho = orthogonalize(right_mps, mpslen)
    # orthogonalize!(L_ortho,1)
    # orthogonalize!(R_ortho,1)
    # normalize!(L_ortho)
    # normalize!(R_ortho)
    

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = Vector{ComplexF64}()

    # Left sweep with truncation 

    Ai = XUinv * L_ortho[1]
    Bi = XVinv * R_ortho[1] 

    # Generalized canonical - no complex conjugation!
    left_env = deltaS * Ai 
    left_env *= Bi 

    u1,s1,v1 = svd(left_env, ind(left_env,1); cutoff=1e-14, maxdim=20)

    linds = []
    full_mps = left_env
    for jj = 2:mpslen
        full_mps *= L_ortho[jj] 
        full_mps *= prime(R_ortho[jj], "Site")
        push!(linds, siteind(L_ortho,jj))
    end

    @show linds
    @show inds(full_mps)


    u2,s2,v2 = svd(full_mps, linds; cutoff=nothing, maxdim=20)

    @show s1
    @show s2

end

""" sweep for generalized truncation, Left(Can)> then <Right(Gen) """
function truncate_normalize_sweep_LR(left_mps::MPS, right_mps::MPS; method::String, svd_cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    L_ortho = orthogonalize(left_mps, mpslen, normalize=false)
    R_ortho = orthogonalize(right_mps,mpslen, normalize=false)

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    # if method == "EIG"
    #     ents_sites = Vector{ComplexF64}()
    # else
    #     ents_sites = Vector{Float64}()
    # end
    
    ents_sites = Vector{Float64}()


    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # No complex conjugation 
        right_env = deltaS * Ai 
        right_env = right_env * Bi 

        @assert order(right_env) == 2

        if method == "EIG"

            U, S, Vdag, trunc_err = eigtrunc(left_env, svd_cutoff, chi_max)

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = (Vdag*delta( inds(Vdag, "v"), inds(U, "u")) *delta( inds(Vdag, "Link"), inds(U, "Link")) ) * isqS  # should be same as inv(U) or pinv(U) * isqS but need to adjust indices
            XUinv = sqS * U

            XV = (U * delta( inds(Vdag, "v"), inds(U, "u"))*delta( inds(Vdag, "Link"), inds(U, "Link"))) * isqS # same as [p]inv(Vdag) * isqS ?
            XVinv = sqS * Vdag

            
        else # default to SVD
            U,S,Vdag = svd(right_env, inds(right_env)[1], cutoff=svd_cutoff, maxdim= chi_max)

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS
            XUinv = sqS * U

            XV = dag(Vdag) * isqS
            XVinv = sqS * Vdag

        end 



        deltaS = delta(inds(S))

        # Set updated matrices
        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #println(S)
        # compute some "effective entropy" using the SVs here
        # gen_ent_cut = 0.
        # for n=1:dim(S,1)
        #     p = S[n,n]        # I don't think we need the ^2 here 
        #     gen_ent_cut -= p * log(p)
        # end

        # Eg renyi 2 should be cheap? just for convergence - always abs() even if it's complex 
        #gen_ent_cut = sum(abs2.(S)/(norm(S)^2))
        #push!(ents_sites, gen_ent_cut)

        # TODO gotta do it the other way round 
        push!(ents_sites, log(sum(S)))


    end

    # the last two 
    A1 = XUinv * L_ortho[1]
    B1 = XVinv * R_ortho[1]

    overlap = deltaS * ( A1 * B1 ) 

    
    @assert order(overlap) == 0 
    # normalize overlap to 1 at each step 
    L_ortho[1] =  A1 /sqrt(overlap[1])
    R_ortho[1] =  B1 /sqrt(overlap[1])

    #println("div by overlap $(overlap)")
    #println(complex(overlap))

    return L_ortho, R_ortho, ents_sites

end


"""
Just bring the MPS to generalized *left* canonical form without truncating (as far as possible)
"""
function gen_canonical_left(in_mps::MPS)
    return truncate_normalize_sweep_sym(in_mps; svd_cutoff=1e-14, chi_max=2*maxlinkdim(in_mps))
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
"""
function gen_canonical_right(in_mps::MPS)
    return truncate_normalize_sweep_sym_right(in_mps; svd_cutoff=1e-14, chi_max=2*maxlinkdim(in_mps))
end








#
# * Symmetric cases - should be eventually superseded by iGensors functions 
# 

""" 
Symmetric case: Truncates a single MPS optimizing overlap (L|L) (no conj)
returns L_ortho, ents_sites
"""
function truncate_normalize_sweep_sym(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100)

    mpslen = length(left_mps)

    # bring to "standard" right canonical forms individually - ortho center on the 1st site 
    #println("LEFT_mps= ")
    #println(left_mps)
    L_ortho = orthogonalize(left_mps,1, normalize=false)

    XUinv= ITensor(1.)

    ents_sites = Vector{Float64}()

    # Start from the operator/final state side (for me that's on the left)
    #vRs = linkinds(L_ortho)
    for ii =1:mpslen-1
        Ai = XUinv * L_ortho[ii]

        vR = linkind(L_ortho,ii)
    
        #left_env = Ai 
        #left_env = left_env * prime(Ai, vR)
        left_env = Ai * prime(Ai, vR)


        @assert order(left_env) == 2

        #println(left_env)
        #U,S = symmetric_svd(left_env, svd_cutoff=svd_cutoff)
        U,S = symmetric_svd_arr(left_env, svd_cutoff=svd_cutoff, chi_max=chi_max)

        # orig version (works)
        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        #XU = transpose(conj(U)) * isqS
        XUinv = sqS * U
        #

        #=new version 

        sqS = sqrt.(diag(S))
        isqS = sqS.^(-1)
        
        XU = dag(U) * diagITensor(isqS.storage.data, inds(S))
        XUinv = diagITensor(sqS.storage.data, inds(S)) * U
        =#

        #L_ortho[ii] =  Ai * XU  #replacetags(Ai * XU , "Link,v", "Link,v=$ii")
        L_ortho[ii] =  replacetags(Ai, "v", "Link,v=$(ii-1)") * replacetags(XU, "v", "Link,v=$ii") 
        push!(ents_sites, log(sum(S)))
    end

    # the last two 
    An = XUinv * L_ortho[mpslen]

    overlap = An * An 

    
    @assert order(overlap) == 0 
    # normalize overlap to 1 at each step 
    L_ortho[mpslen] =  replacetags(An, "v", "Link,v=$(mpslen-1)") /sqrt(overlap[1])


    return L_ortho, ents_sites

end



""" More aggressive version (carrying around left_env which slows everything down though ) 
 First right-orthogonalizes, then performs a generalized left sweep truncating """ 
function truncate_normalize_sweep_sym!(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100)

    mpslen = length(left_mps)

    orthogonalize!(left_mps,1, normalize=false)

    XUinv= ITensor(1.)
    left_env = ITensor(1.)
    Ai = ITensor(1.)

    ents_sites = Vector{Float64}()

    for ii =1:mpslen-1

        Ai = XUinv * left_mps[ii]

        left_env = left_env * Ai
        left_env = left_env * Ai'
        left_env *= delta(siteind(left_mps,ii),siteind(left_mps,ii)')

        U,S = symmetric_svd_arr(left_env, svd_cutoff=svd_cutoff, chi_max=chi_max)

        #slightly faster version (check)

        sqS = sqrt.(diag(S))
        isqS = sqS.^(-1)
        
        XU = dag(U) * diagITensor(isqS.storage.data, inds(S))
        XUinv = diagITensor(sqS.storage.data, inds(S)) * U
        #

        left_mps[ii] =  Ai * XU  #replacetags(Ai * XU , "Link,v", "Link,v=$ii")

        left_env =  left_mps[ii] * prime(left_mps[ii], inds(left_mps[ii])[end])
        #println(inds(left_env))
        #println("###")
        #println(linkind(left_mps,ii))
        #println("###")


        push!(ents_sites, log(sum(S)))
    end

    # the last two 
    An = XUinv * left_mps[mpslen]

    overlap = An * An 

    # normalize overlap to 1 at each step 
    left_mps[mpslen] =  An /sqrt(overlap[1])


    return ents_sites

end



function truncate_normalize_sweep_sym_ite!(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100, method="gen_one")
    orthogonalize!(left_mps,1)
    _, _, ents = orthogonalize_gen_ents!(left_mps, length(left_mps); cutoff=svd_cutoff, normalize=true, method)
    return ents
end




""" 
Symmetric case sweep to bring to gen. RIGHT canonical form
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





""" Check that two MPS are in (generalized-symmetric) *left* canonical form """
function check_gencan_left_sym_phipsi(psi::MPS, phi::MPS, verbose::Bool=false)

    @assert length(psi) == length(phi)

    if verbose
        println("Checking LEFT gen/sym form")
    end
    
    if abs(overlap_noconj(psi,phi) - 1.) > 1e-7 || abs(1. - scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] ))) > 1e-7
        println("overlap = $(overlap_noconj(psi,phi))), alt = $(scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] )))")
    end

    mpslen = length(psi)
    # Start from the operator/final state side (for me that's on the left)
    left_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[1:end-1], phi[1:end-1]))
        left_env =  left_env * Ai
        left_env = left_env * prime(Bi, commoninds(Bi,linkinds(phi)))    #* delta(wLa, wLb) )
        @assert order(left_env) == 2
        if norm(array(left_env)- diagm(diag(array(left_env)))) > 0.1
            @warn("[R]non-diag@[$ii]")
        end
        delta_norm = norm(array(left_env) - I(size(left_env)[1])) 
        if delta_norm > 0.0001
            @warn("[R]non-can@[$ii], $delta_norm")
        end
    end
    if verbose
        @info("Done checking RIGHT gen/sym form")
    end
    
end


""" Check that two MPS are in (generalized-symmetric) *right* canonical form
TODO not implemented yet  """
function check_gencan_right_sym_phipsi(psi::MPS, phi::MPS, verbose::Bool=false)
    if verbose
        @info("Checking RIGHT gen/sym form")
    end
    
    if abs(overlap_noconj(psi,phi) - 1.) > 1e-7 || abs(1. - scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] ))) > 1e-7
        println("overlap = $(overlap_noconj(psi,phi))), alt = $(scalar(psi[end]*phi[end]*delta(linkinds(psi)[end],linkinds(phi)[end] )))")
    end

    mpslen = length(psi)
    # Start from the initial state side (for me that's on the right)
    right_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[end:-1:1], phi[end:-1:1]))
        right_env =  right_env * Ai
        right_env = right_env * prime(Bi, commoninds(Bi,linkinds(phi)))    #* delta(wLa, wLb) )
        @assert order(right_env) == 2
        if norm(array(right_env)- diagm(diag(array(right_env)))) > 0.1
            @warn("[R]non-diag@[$ii]")
        end
        delta_norm = norm(array(right_env) - I(size(right_env)[1])) 
        if delta_norm > 0.01
            @warn("[R]non-can@[$ii], $delta_norm")
        end
    end
    if verbose
        @info("Done checking RIGHT gen/sym form")
    end
    
end







""" Check that an MPS is in (generalized-symmetric) left canonical form,
ie. that when we contract everything from the left to the right we get identities
Should be superseded by check_gen_ortho() in my iGensors """
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

