""" Builds left and right envs from a Columns struct. Returns left_envs, right_envs """
function initialize_envs_rdm(cc::Columns, trunc_params)

    (; cutoff, maxdim) =  trunc_params

    NN = length(cc)

    ll =  cc[1]
    rr =  cc[NN]

   left_envs =  Environments(NN, ll, :L)  # Sets L[1]
   right_envs = Environments(NN, rr, :R)  # Sets R[NN-1]

   p = Progress(2*NN-2; dt=20, showspeed=true)

   #  Li+1 = Li * Ti+1  ;  Ri = Ti+1 * Ri+1
   for jj in 1:NN-2

        @debug "Building L[$(jj+1)] = LL[$(jj)]*E[$(jj+1)]"
  
        ll = applyns(cc[jj+1], left_envs[jj]; truncate=true, cutoff, maxdim=maxdim)

        update_env!(left_envs, jj+1, ll)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(ll))]")])

    end

    for jj = NN-1:-1:2

        @debug "Building R[$(jj-1)] = E[$(jj)]*R[$(jj)]"
    
        #@show length(rr), length(mpo_R)
        rr = applyn(cc[jj], right_envs[jj]; truncate=true, cutoff, maxdim=maxdim)
        #@show rr

        update_env!(right_envs, jj-1, rr)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(rr))]")])

   end

   return left_envs, right_envs

end
