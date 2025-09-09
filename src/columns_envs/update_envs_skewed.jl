""" Sweep rebuilding adjacent L-R environments using RTM. 
Attempt at letting bond dimension grow if necessary: instead of eg. taking Li, build Li+1 and update 
    with Ri+1, we take Li and Ri+2, build Li+1 and Ri+i and truncate over those. """
function sweep_rebuild_envs_rtm_skewed!(left_envs::Environments, right_envs::Environments, cc::Columns, 
    truncp::TruncParams; verbose::Bool=false)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

    ll = cc[1]

    left_envs.norms[1] = norm(ll)
    ll = normalize(ll)
    left_envs[1] = ll

    # Update Left envs using current {right_envs} as input
    #  L[jj] = L[jj-1] * E[jj]

    for jj in 2:NN-2 # TODO Need to update ll[N-1] separately 

        ll = applyns(cc[jj], left_envs[jj-1]; truncate=false)
        rr = applyn(cc[jj+1], right_envs[jj+1])
        ll, _, _ = truncate_rsweep(ll, rr, truncp; fast=true)

        update_env!(left_envs, jj, ll)
        # ll = orthogonalize(ll, length(ll))
        # left_envs.norms[jj] = norm(ll)
        # ll = normalize(ll)
        # left_envs[jj] = ll

        if verbose
            @info "updating L[$(jj-1)]E[$(jj)] = L[$(jj)] with R[$(jj)]"
        end
    end

    # Last one 
    ll = applyns(cc[NN-1], left_envs[NN-2]; truncate=false)
    ll, _, _ = truncate_rsweep(ll, right_envs[NN-1], truncp; fast=true)

    update_env!(left_envs, NN-1, ll)


    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-1:-1:3 # TODO update R[1] separately 

        rr = applyn(cc[jj], right_envs[jj])
        ll = applyns(cc[jj-1], left_envs[jj-2])
        _, rr, _ = truncate_rsweep(ll, rr, truncp, fast=true)

        update_env!(right_envs, jj-1, rr)

        if verbose
            @info "updating E[$(jj)]R[$(jj)] = R[$(jj-1)] with L[$(jj-1)] = L[$(jj-2)]E[$(jj-1)]"
        end
    end

    rr = applyn(cc[2], right_envs[2])
    _, rr, _ = truncate_rsweep(left_envs[1], rr, truncp, fast=true)
    update_env!(right_envs, 1, rr)


    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end
