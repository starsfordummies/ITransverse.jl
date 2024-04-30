""" If we're not sure that our generalized Left env in the sweep is normalized to 1,
 we can drag it along and multiply to build our environment at every step """
function truncate_sweep_keep_lenv(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    #@show ortho_lims(left_mps)
    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)
    #@show ortho_lims(L_ortho)


    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 
    left_prev = ITensor(1.)

    ents_sites = Vector{ComplexF64}()

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        #left_env = deltaS
        left_env = left_prev
        left_env *= Ai 
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


        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        left_prev *= L_ortho[ii]
        left_prev *= R_ortho[ii]
        #left_prev *= deltaS



        deltaS = delta(inds(S))


        #  if ii > 2 
        # #     #@show inds(S)
        # #     #@info "$ii $(norm(left_prev - deltaS)) $(norm(S))"
        #     if norm(S) > 10000
        #          @show S
        #          @show [norm_gen.(a for a in left_mps)]
        #          @show [norm_gen.(a for a in right_mps)]
        #          sleep(5)
        #     end
        #  end

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


""" Here we normalize the left environment at every step, 
    so that truncation should be more uniform
     (need to check we're not destroying anyhting though """
function truncate_sweep_aggressive_normalize(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    #@show ortho_lims(left_mps)
    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)
    #@show ortho_lims(L_ortho)


    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 
    #left_prev = ITensor(1.)

    ents_sites = Vector{ComplexF64}()

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = deltaS
        #left_env = left_prev
        left_env *= Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        if method == "SVD" 
            #@info "[$ii] renormalizing left_env from $(norm(left_env))"

            lnorm = norm(left_env)
            left_env /= lnorm
            #normalize!(left_env)
            
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


        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #left_prev *= L_ortho[ii]
        #left_prev *= R_ortho[ii]
        #left_prev *= deltaS

        ratnorm = norm(L_ortho[ii])/norm(R_ortho[ii])
        L_ortho[ii] /= sqrt(lnorm*ratnorm)
        R_ortho[ii] /= sqrt(lnorm/ratnorm)


        deltaS = delta(inds(S))


        #  if ii > 2 
        # #     #@show inds(S)
        # #     #@info "$ii $(norm(left_prev - deltaS)) $(norm(S))"
        #     if norm(S) > 10000
        #          @show S
        #          @show [norm_gen.(a for a in left_mps)]
        #          @show [norm_gen.(a for a in right_mps)]
        #          sleep(5)
        #     end
        #  end

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
