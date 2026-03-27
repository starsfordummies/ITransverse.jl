""" Sweep rebuilding adjacent L-R environments using RTM.
Recall rules:
-  L[jj] = L[jj-1] * E[jj] 
-  R[jj] = E[jj-1] * R[jj-1] 
"""
function sweep_rebuild_envs_rtm!(left_envs::Environments, right_envs::Environments, cc::Columns, truncp; edge_buffer::Int=1, kwargs...)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

  
    @debug "Setting L[1] = E[1])"
    update_env!(left_envs, 1, cc[1]; kwargs...)

    @debug "Setting R[$(NN-1)] = E[$(NN)])"
    update_env!(right_envs, NN-1, cc[NN]; kwargs...)

    # If we want a buffer of un-truncated environments at the boundaries 
    for jj = 2:edge_buffer
        @debug "Setting L[$(jj)] = L[$(jj-1)]E[$(jj)])"
        update_env!(left_envs, jj, applyns(cc[jj], left_envs[jj-1]; truncate=false))

        @debug "Setting R[$(NN-jj)] = E[$(NN-jj+1)]R[$(NN-jj+1)]"
        update_env!(right_envs, NN-jj, applyn(cc[NN-jj+1], right_envs[NN-jj+1]; truncate=false))

    end

    for jj in edge_buffer+1:NN-1

        @debug "updating L[$(jj)] via overlap (L[$(jj-1)]E[$(jj)] | R[$(jj)])"
        ll, _ = tlapply(left_envs[jj-1], cc[jj], right_envs[jj]; truncp...)
        update_env!(left_envs, jj, ll; kwargs...)

    end

    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-edge_buffer:-1:2
        @debug "updating R[$(jj-1)] via overlap (L[$(jj-1)]|E[$(jj)]R[$(jj)])"
        _, rr = trapply(left_envs[jj-1], cc[jj], right_envs[jj]; truncp...)
        update_env!(right_envs, jj-1, rr; kwargs...)
    end

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end


""" Sweep rebuilding adjacent L-R environments using RTM. 
At each site i we take Li and Ri+2, build L[i+1]= L[i]E[i] and R[i+i] = E[i+1]R[i+2] and truncate over their overlap.
This should allow for the bond dimension to grow as necessary """
function sweep_rebuild_envs_rtm_twocol!(left_envs::Environments, right_envs::Environments, cc::Columns, truncp; kwargs...)

    NN = length(cc)
    @assert length(left_envs) == length(right_envs) == NN-1

    @debug "updating L[1] = E[1]"
    update_env!(left_envs, 1, cc[1]; kwargs...)

    # Left envs update using current right_envs as input,  L[jj] = L[jj-1] * E[jj]

    for jj in 2:NN-2  
        @debug "updating L[$(jj)] from opt. overlap (L[$(jj-1)]E[$(jj)] | E[$(jj+1)]R[$(jj+1)])"
        ll, _ = tlrapply(left_envs[jj-1], cc[jj], cc[jj+1], right_envs[jj+1]; truncp...)
        update_env!(left_envs, jj, ll; kwargs...)
    end

    # Last one 
    @debug "updating L[$(NN-1)] from opt. overlap (L[$(NN-2)]E[$(NN-1)] | R[$(NN-1)])"
    ll, _ = tlapply(left_envs[NN-2], cc[NN-1], right_envs[NN-1]; truncp...)
    update_env!(left_envs, NN-1, ll; kwargs...)

    @debug "updating R[$(NN-1)] = E[$(NN)]"
    update_env!(right_envs, NN-1, cc[NN]; kwargs...)

    # Update Right envs using {left_envs},  R[jj-1] = E[jj] * R[jj]  
    for jj in NN-2:-1:2 
        @debug "updating R[$(jj)] from overlap  (L[$(jj-1)]E[$(jj)] | E[$(jj+1)]R[$(jj+1)])"
        _, rr = tlrapply(left_envs[jj-1], cc[jj], cc[jj+1], right_envs[jj+1]; truncp...)
        update_env!(right_envs, jj, rr; kwargs...)
    end

    @debug "updating R[1] from overlap  (L[1] | E[2]R[2])"
    _, rr = trapply(left_envs[1], cc[2], right_envs[2]; truncp...)
    update_env!(right_envs, 1, rr; kwargs...)

    return max(maxlinkdim(left_envs),maxlinkdim(right_envs))

end
