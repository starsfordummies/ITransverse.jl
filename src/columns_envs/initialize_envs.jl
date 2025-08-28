""" Builds left and right envs from a Columns struct. Returns left_envs, right_envs """
function initialize_envs_rdm(cc::Columns, trunc_params; verbose::Bool=false)

    (; cutoff, maxbondim) =  trunc_params

    NN = length(cc)

    ll =  cc[1]
    rr =  cc[NN]

   left_envs =  Environments(NN, ll, "L")  # Sets L[1]
   right_envs = Environments(NN, rr, "R")  # Sets R[NN-1]

   p = Progress(NN-2; dt=20, showspeed=true)

   #  Li+1 = Li * Ti+1  ;  Ri = Ti+1 * Ri+1
   for jj in 1:NN-2

        if verbose 
            @info "Building L[$(jj+1)] = LL[$(jj)]*E[$(jj+1)]"
            @info "Building R[$(NN-jj-1)] = E[$(NN-jj)]*R[$(NN-jj)]"
        end

        mpo_L = cc[jj+1]
        ll = applyns(mpo_L, ll; truncate=true, cutoff, maxdim=maxbondim)
        ll = orthogonalize(ll, length(ll))
        left_envs.norms[jj+1] = norm(ll)
        ll = normalize(ll)
        left_envs[jj+1] = ll

        mpo_R = cc[NN-jj]
        #@show length(rr), length(mpo_R)
        rr = applyn(mpo_R, rr; truncate=true, cutoff, maxdim=maxbondim)
        #@show rr

        rr = orthogonalize(rr, length(rr))
        right_envs.norms[NN-jj-1] = norm(rr)
        rr = normalize(rr)
        right_envs[NN-jj-1] = rr 

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(ll))]")])

   end

   return left_envs, right_envs

end
