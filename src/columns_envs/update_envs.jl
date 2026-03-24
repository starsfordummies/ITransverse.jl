""" Sweep rebuilding adjacent L-R environments using RTM  """
function sweep_rebuild_envs_rtm!(left_envs::Environments, right_envs::Environments, cc::Columns, truncp)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

    update_env!(left_envs, 1, cc[1])

    for jj in 2:NN-1

        ll = applyns(cc[jj], left_envs[jj-1]; truncate=false)
        ll, _, _ = truncate_sweep(ll, right_envs[jj]; truncp...) # direction = :right

        update_env!(left_envs, jj, ll)

        @debug "updating L[$(jj-1)]E[$(jj)] = L[$(jj)] with R[$(jj)]"
        
    end

    update_env!(right_envs, NN-1, cc[NN])

    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-1:-1:2

        rr = applyn(cc[jj], right_envs[jj])
        _, rr, _ = truncate_sweep(left_envs[jj-1], rr; truncp...) # direction = :right

        update_env!(right_envs, jj-1, rr)

        @debug "updating E[$(jj)]R[$(jj)] = R[$(jj-1)] with L[$(jj-1)]"

    end

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end


""" Sweep rebuilding adjacent L-R environments using RTM. 
Attempt at letting bond dimension grow if necessary: instead of eg. taking Li, build Li+1 and update 
    with Ri+1, we take Li and Ri+2, build Li+1 and Ri+i and truncate over those. """
function sweep_rebuild_envs_rtm_twocol!(left_envs::Environments, right_envs::Environments, cc::Columns, truncp)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

    update_env!(left_envs, 1, cc[1])

    # Update Left envs using current {right_envs} as input
    #  L[jj] = L[jj-1] * E[jj]

    for jj in 2:NN-2 # TODO Need to update ll[N-1] separately 

        ll = applyns(cc[jj], left_envs[jj-1]; truncate=false)
        rr = applyn(cc[jj+1], right_envs[jj+1])
        ll, _, _ = truncate_sweep(ll, rr; truncp...) # direction = right

        update_env!(left_envs, jj, ll)
    
        @debug "updating L[$(jj-1)]E[$(jj)] = L[$(jj)] with R[$(jj)]"
   
    end

    # Last one 
    ll = applyns(cc[NN-1], left_envs[NN-2]; truncate=false)
    ll, _, _ = truncate_sweep(ll, right_envs[NN-1]; truncp)

    update_env!(left_envs, NN-1, ll)


    update_env!(right_envs, NN-1, cc[NN])

    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-1:-1:3 

        rr = applyn(cc[jj], right_envs[jj])
        ll = applyns(cc[jj-1], left_envs[jj-2])
        _, rr, _ = truncate_sweep(ll, rr; truncp...)

        update_env!(right_envs, jj-1, rr)

        @debug "updating E[$(jj)]R[$(jj)] = R[$(jj-1)] with L[$(jj-1)] = L[$(jj-2)]E[$(jj-1)]"
    end

    rr = applyn(cc[2], right_envs[2])
    _, rr, _ = truncate_sweep(left_envs[1], rr; truncp...)
    update_env!(right_envs, 1, rr)

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end
