""" Updates (RTM) the adjacent environments after a change at position `jj`, using T_mpo = T[jj]
 - If `sweep_dir = "L"`, updates L[jj] = L[jj-1] * T[jj]
 - If `sweep_dir = "R"`, updates R[jj-1] = T[jj] * R[jj]  
 """
function sweep_rebuild_envs_rtm!(left_envs::Environments, right_envs::Environments, cc::Columns, 
    truncp::TruncParams; verbose::Bool=false)

    NN = length(cc)

    ll = cc[1]

    left_envs.norms[1] = norm(ll)
    ll = normalize(ll)
    left_envs[1] = ll


    # Update Left envs using current {right_envs} as input
    for jj in 2:NN-2

        mpo_L =  cc[jj]
        ll = applyns(mpo_L, ll; truncate=false)
        ll, _, _ = truncate_rsweep(ll, right_envs[jj], truncp; fast=true)

        ll = orthogonalize(ll, length(ll))
        left_envs.norms[jj] = norm(ll)
        ll = normalize(ll)
        left_envs[jj] = ll

        if verbose
            @info "updating L[$(jj-1)]E[$(jj)] = L[$(jj)] with R[$(jj)]"
        end
    end

    rr =  cc[end] 
    right_envs.norms[end] = norm(rr)
    rr = normalize(rr)
    right_envs[end] = rr
  
    # Update Right envs using  current {left_envs} as input

    for jj in NN-1:-1:2
        mpo_j = cc[jj]

        rr = applyn(mpo_j, rr)
        _, rr, _ = truncate_rsweep(left_envs[jj-1], rr, truncp, fast=true)

        orthogonalize!(rr,length(rr))
        right_envs.norms[jj-1] = norm(rr)
        normalize!(rr)
        right_envs[jj-1] = rr 
        #@info "updating R[$(jj)]"
        if verbose
            @info "updating E[$(jj)]R[$(jj)] = R[$(jj-1)] with L[$(jj-1)]"
        end

    end

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end

