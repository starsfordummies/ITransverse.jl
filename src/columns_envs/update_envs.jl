""" Sweep rebuilding adjacent L-R environments using RTM  """
function sweep_rebuild_envs_rtm!(left_envs::Environments, right_envs::Environments, cc::Columns, 
    truncp::TruncParams; verbose::Bool=false)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

    update_env!(left_envs, 1, cc[1])

    # ll = cc[1]
    # left_envs.norms[1] = norm(ll)
    # ll = normalize(ll)
    # left_envs[1] = ll

    # Update Left envs using current {right_envs} as input
    #  L[jj] = L[jj-1] * E[jj]

    for jj in 2:NN-1

        ll = applyns(cc[jj], left_envs[jj-1]; truncate=false)
        ll, _, _ = truncate_rsweep(ll, right_envs[jj], truncp; fast=true)

        update_env!(left_envs, jj, ll)
        # ll = orthogonalize(ll, length(ll))
        # left_envs.norms[jj] = norm(ll)
        # ll = normalize(ll)
        # left_envs[jj] = ll

        if verbose
            @info "updating L[$(jj-1)]E[$(jj)] = L[$(jj)] with R[$(jj)]"
        end
    end

    update_env!(right_envs, NN-1, cc[NN])

    # rr =  cc[NN] 
    # right_envs.norms[NN-1] = norm(rr)
    # rr = normalize(rr)
    # right_envs[NN-1] = rr
  
    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-1:-1:2

        rr = applyn(cc[jj], right_envs[jj])
        _, rr, _ = truncate_rsweep(left_envs[jj-1], rr, truncp, fast=true)

        update_env!(right_envs, jj-1, rr)

        # orthogonalize!(rr,length(rr))
        # right_envs.norms[jj-1] = norm(rr)
        # normalize!(rr)
        # right_envs[jj-1] = rr 

        if verbose
            @info "updating E[$(jj)]R[$(jj)] = R[$(jj-1)] with L[$(jj-1)]"
        end

    end

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end

