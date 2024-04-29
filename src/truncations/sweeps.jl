
function truncate_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    #@show ortho_lims(left_mps)
    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)
    #@show ortho_lims(L_ortho)


    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = Vector{ComplexF64}()

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = Ai * deltaS 
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
            
            #U,S,Vdag = svd(left_env, ind(left_env,1); cutoff=cutoff^2, maxdim=chi_max, use_absolute_cutoff=true)
            #U,S,Vdag = svd(left_env, ind(left_env,1); cutoff=nothing, maxdim=chi_max, use_absolute_cutoff=true)
            U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)


            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS
            XUinv = sqS * U

            XV = dag(Vdag) * isqS
            XVinv = sqS * Vdag


        else
            throw(ArgumentError("need to specify method EIG|SVD - current method=$method"))
        end

        deltaS = delta(inds(S))

        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #! This could be nasty if we have imaginary stuff...
        #! should not be a problem for SVDs though
        push!(ents_sites, log(sum(S)))
      
    end

    # the last two 
    L_ortho[end] = XUinv * L_ortho[end]
    R_ortho[end] = XVinv * R_ortho[end]

    gen_overlap = deltaS * ( L_ortho[end] *  R_ortho[end] ) 

    return L_ortho, R_ortho, ents_sites, gen_overlap

end



"""
Truncates optimizing the overlap (L|R), returns new L and R in generalized orthogonal form 
and (an estimate for) the gen. entropy at each site.
Cutoff on SV of the transition matrix, given by `cutoff` param

For this, we first bring to the usual *right* canonical form both MPS (ortho center on 1st site), \\
then we build environments L|R from the *left* and truncate on their SVDs (or EIGs depending on `method`)

So this can be seen as a "RL: Right(can)Left(gen)" sweep 
"""
function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS, truncp::trunc_params)
    truncate_normalize_sweep(left_mps, right_mps, method=truncp.ortho_method, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
end

function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    L_ortho, R_ortho, ents_sites, gen_overlap = truncate_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    L_ortho = normbyfactor(L_ortho, sqrt(gen_overlap))
    R_ortho = normbyfactor(R_ortho, sqrt(gen_overlap))

    return L_ortho, R_ortho, ents_sites

end


function _truncate_normalize_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    # bring to "standard" right canonical forms individually - ortho center on the 1st site 
    # making copies along the way 

    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)

    # # ! does this change anything ? doesn't seem like it 
    # L_ortho = orthogonalize(left_mps,  mpslen)
    # R_ortho = orthogonalize(right_mps, mpslen)
    # orthogonalize!(L_ortho,1)
    # orthogonalize!(R_ortho,1)
    #normalize!(L_ortho)
    #normalize!(R_ortho)
    

    # overlap = overlap_noconj(L_ortho,R_ortho)
    # @info "normalizing by overlap $(overlap)"

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    #@show XUinv.tensor.storage.data

    ents_sites = Vector{ComplexF64}()

    # Left sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = Ai * deltaS 
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
            
            U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)
            #U,S,Vdag = svd(left_env, ind(left_env,1); cutoff=cutoff^2, maxdim=chi_max, use_absolute_cutoff=true)

            #@show sum(S), sum(S.^2)

            # if sum(S) > 10. 
            #     throw(ArgumentError(" bad $(diag(matrix(S)))"))
            # end

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS
            XUinv = sqS * U

            XV = dag(Vdag) * isqS
            XVinv = sqS * Vdag


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
    # @assert order(overlap) == 0 
    
    # # normalize overlap to 1 at each step 
    # L_ortho[end] =  An/sqrt(scalar(overlap))
    # R_ortho[end] =  Bn/sqrt(scalar(overlap))

    # check that we're in the appropriate gen ortho form at the end 
    #check_gencan_left_sym_phipsi(L_ortho, R_ortho)

    #println("div by overlap $(overlap)")
    #println(complex(overlap))

    return L_ortho, R_ortho, ents_sites

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
