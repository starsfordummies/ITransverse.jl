using CUDA 

""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation """
function gpu_truncate_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    #@show ortho_lims(left_mps)
    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)
    #@show ortho_lims(L_ortho)


    XUinv, XVinv, deltaS = (NDTensors.cu(ITensor(1.)), NDTensors.cu(ITensor(1.)), NDTensors.cu(ITensor(1.))) 

    ents_sites = Vector{ComplexF64}()

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = Ai * deltaS 
        left_env *= Bi 

        #@assert order(left_env) == 2

        U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)

        # WORKAROUND dense(S) until ITensors fix it 
        sqS = sqrt.(dense(S))
        isqS = sqS.^(-1)
        
        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag


        deltaS = NDTensors.cu(delta(inds(S)))

        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #! This could be nasty if we have imaginary stuff...
        #! should not be a problem for SVDs though
        #push!(ents_sites, log(sum(S)))
      
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
function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS, truncp::trunc_params)
    truncate_normalize_sweep(left_mps, right_mps, method=truncp.ortho_method, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
end

function truncate_normalize_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    L_ortho, R_ortho, ents_sites, gen_overlap = truncate_sweep(left_mps, right_mps; method, cutoff, chi_max)

    L_ortho = normbyfactor(L_ortho, sqrt(gen_overlap))
    R_ortho = normbyfactor(R_ortho, sqrt(gen_overlap))

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





