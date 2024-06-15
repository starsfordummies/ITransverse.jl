using CUDA 

""" Basic truncate sweep: first brings to regular RIGHT ortho form,
then performs a LEFT generalized canonical sweep with SVD/EIG truncation """
function gpu_truncate_sweep(left_mps::MPS, right_mps::MPS; cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    #@show ortho_lims(left_mps)
    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)
    #@show ortho_lims(L_ortho)


    XUinv, XVinv, deltaS = (NDTensors.cu(ITensor(1.)), NDTensors.cu(ITensor(1.)), NDTensors.cu(ITensor(1.))) 

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = Ai * deltaS 
        left_env *= Bi 

        #@assert order(left_env) == 2

        U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)

        # WORKAROUND CPU-back it until ITensors fix it 
        sqS = NDTensors.cu(sqrt.(NDTensors.cpu(S)))
        isqS = NDTensors.cu(sqS.^(-1))
        
        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag


        deltaS = delta(inds(S))

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

    return L_ortho, R_ortho, gen_overlap

end
