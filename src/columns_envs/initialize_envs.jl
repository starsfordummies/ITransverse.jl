""" Builds left and right envs from a Columns struct. Returns left_envs, right_envs """
function initialize_envs_rdm(cc::Columns, trunc_params=(cutoff=1e-12, maxdim=16); edge_buffer::Int=1)

    NN = length(cc)

    ll =  cc[1]
    rr =  cc[NN]

    @debug "Initialize L[1]"
    left_envs =  Environments(NN, ll, :L)  # Sets L[1]
    
    @debug "Initialize R[$(NN-1)]"
    right_envs = Environments(NN, rr, :R)  # Sets R[NN-1]

    p = Progress(2*NN-2; dt=20, showspeed=true)

    #  Li+1 = Li * Ti+1  ;  Ri = Ti+1 * Ri+1

    # Left buffer
    for jj in 1:edge_buffer 

        @debug "Building L[$(jj+1)] = LL[$(jj)]*E[$(jj+1)] (no trunc)"

        ll = applyns(cc[jj+1], left_envs[jj]; truncate=false)
        update_env!(left_envs, jj+1, ll)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(ll))]")])

    end

    # Left envs 
    for jj in edge_buffer+1:NN-2

        @debug "Building L[$(jj+1)] = LL[$(jj)]*E[$(jj+1)]"

        ll = applyns(cc[jj+1], left_envs[jj]; trunc_params...)
        update_env!(left_envs, jj+1, ll)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(ll))]")])

    end

    # Right buffer 
    for jj = NN-1:-1:NN-edge_buffer

        @debug "Building R[$(jj-1)] = E[$(jj)]*R[$(jj)] (no trunc)"

        rr = applyn(cc[jj], right_envs[jj]; truncate=false)
        update_env!(right_envs, jj-1, rr)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(rr))]")])

    end

    # Right envs
    for jj = NN-1-edge_buffer:-1:2

        @debug "Building R[$(jj-1)] = E[$(jj)]*R[$(jj)]"

        rr = applyn(cc[jj], right_envs[jj]; trunc_params...)
        update_env!(right_envs, jj-1, rr)

        next!(p; showvalues = [(:Info,"[RDM Init envs][$(jj)][χ=$(maxlinkdim(rr))]")])

    end

    return left_envs, right_envs

    end
