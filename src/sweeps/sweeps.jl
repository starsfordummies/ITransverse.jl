""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation """

function truncate_lsweep(right_mps::MPS, left_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    R_ortho = orthogonalize(right_mps, 1)
    L_ortho = orthogonalize(left_mps,  1)

    XUinv, XVinv, left_env = (ITensor(1.), ITensor(1.), ITensor(1.)) 

    ents_sites = ComplexF64[]

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

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

            XV = (U * delta( ind_v, ind_u)*delta( link_v, link_u )) * isqS 
            #XV = (U * delta( inds(Vdag, "v"), inds(U, "u"))*delta( inds(Vdag, "Link"), inds(U, "Link"))) * isqS # same as [p]inv(Vdag) * isqS ?
            XVinv = sqS * Uinv

        elseif method == "SVD" 
            
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

        push!(ents_sites, log(sum(S)))
      
    end

    # the last two 
    L_ortho[end] = XUinv * L_ortho[end]
    R_ortho[end] = XVinv * R_ortho[end]

    gen_overlap = scalar(deltaS * ( L_ortho[end] *  R_ortho[end] ) )

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


function truncate_normalize_rsweep(right_mps::MPS, left_mps::MPS, truncp::trunc_params)
    truncate_rsweep(right_mps, left_mps, method=truncp.ortho_method, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    LR =  overlap_noconj(right_mps,left_mps)
    right_mps[1] /= sqrt(LR)
    left_mps[1] /= sqrt(LR)
end

""" Brings to right generalized canonical form two MPS, truncating along the way if necessary """
function truncate_rsweep(right_mps::MPS, left_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    R_ortho = orthogonalize(right_mps,mpslen, normalize=false)
    L_ortho = orthogonalize(left_mps, mpslen, normalize=false)

    XUinv, XVinv, right_env = (ITensor(1.), ITensor(1.), ITensor(1.)) 
    
    ents_sites = ComplexF64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * R_ortho[ii]
        Bi = XVinv * L_ortho[ii] 

        # No complex conjugation 
        right_env *= Ai 
        right_env *= Bi 

        rnorm = norm(right_env)
        right_env /= rnorm
        

        @assert order(right_env) == 2

        if method == "EIG"

            U, S, Vdag, trunc_err = eigtrunc(right_env, cutoff, chi_max)

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = (Vdag*delta( inds(Vdag, "v"), inds(U, "u")) *delta( inds(Vdag, "Link"), inds(U, "Link")) ) * isqS  # should be same as inv(U) or pinv(U) * isqS but need to adjust indices
            XUinv = sqS * U

            XV = (U * delta( inds(Vdag, "v"), inds(U, "u"))*delta( inds(Vdag, "Link"), inds(U, "Link"))) * isqS # same as [p]inv(Vdag) * isqS ?
            XVinv = sqS * Vdag

            
        else # default to SVD
            U,S,Vdag = svd(right_env, inds(right_env)[1]; cutoff=cutoff, maxdim=chi_max)

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS
            XUinv = sqS * U

            XV = dag(Vdag) * isqS
            XVinv = sqS * Vdag

        end 

        # Set updated matrices
        R_ortho[ii] = Ai * XU  
        L_ortho[ii] = Bi * XV

        push!(ents_sites, log(sum(S)))

    end

    # the last two 
    R_ortho[1] = XUinv * R_ortho[1]
    L_ortho[1] = XVinv * L_ortho[1]

    return R_ortho, L_ortho, ents_sites

end
