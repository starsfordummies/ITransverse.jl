

########################################################
######## trcontract ####################################
########################################################

""" Applies AR to ψR and truncates building RTM from |AψR><ψL| """ 
function trcontract(::Algorithm"RTM",
        ψL::MPS,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        compute_ov_before::Bool = true, # we actually do it anyway 
        direction = :right,
        kwargs...,
    )
    if direction == :right
        L, R, sv, ovb = _trcontract_rtm_right(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        L, R, sv, ovb = _trcontract_rtm_left(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end

    if !compute_ov_before
        ovb = NaN
    end
    
    return TruncLR(L, R, sv, ovb, NaN)
end



""" Builds LEFT environments, sweeps RIGHT→LEFT """
function _trcontract_rtm_left(ψL::MPS, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    ### Indices prime contractions conventions
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    # N=NR should be the longest 
    # We apply to the right, so: 
    # - nL is what actually determines our out-MPS length
    # - if nL = N > nR, we simply identify ψR[i>nR] = AR[i]
    # - if nL < N = nR, we contract R = ψR[i>nL]*AR[i]
    nL = length(ψL)
    nR = length(ψR)
    N = length(AR)

    @show nL, N, nR

    @assert N >= nR
    @assert N >= nL

    ψLp = ψL'
    ψLpp = prime(siteinds, ψLp)

    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = MPS(nL)
    ψL_out           = MPS(nL)
    S_all            = zeros(Float64, N-1, requested_maxdim)

    # Step 1: build left environments E[j] = sites 1..j
    E   = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env   = env * get(ψR,j) * AR[j] * get(ψLp,j)
        E[j]  = env
        @assert ndims(env) < 4 "env[$(j)] ? $(inds(env))"
    end

    ov_before = scalar(E[N])


    # Step 2: right boundary, absorbing excess AR sites beyond N

    R = ITensors.OneITensor()
    for j = reverse(nL:N)
        R *= get(ψR,j) * AR[j] 
    end
    L = ψLpp[nL]

    r_renorm = nothing

    # Step 3: sweep right → left
    for j in reverse(1:nL-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        rho = E[j] * L * R
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = preserve_mps_tags ? tags(linkind(ψR, j)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? tags(linkind(ψL, j)) : "Link,l=$(j)"

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)
        F   = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        r_renorm = F.u

        @show inds(rho)
        @show j, Ris 

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = dag(U) * R * get(ψR,j) * AR[j] 
        L = dag(V) * L * get(ψLpp,j)

        Svec = collect(storage(S).data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    @show inds(R)
    @show inds(L)

    ψR_out[1] = R
    ψL_out[1] = L

    @show check_mps_sanity(ψL_out)
    @show check_mps_sanity(ψR_out)

    return ψL_out, ψR_out, S_all, ov_before
end

""" Builds RIGHT environments, sweeps LEFT→RIGHT """
function _trcontract_rtm_right(ψL::MPS, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    N  = length(ψL)
    nR = length(ψR)
    NR = length(AR)

    @assert NR >= nR

    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = typeof(ψR)(N)
    ψL_out           = typeof(ψL)(N)
    ψLp              = prime(siteinds, ψL')
    S_all            = zeros(Float64, N-1, requested_maxdim)

    # Step 1: build right environments E[j] = sites j..N (+ excess AR beyond N)
    E = Vector{ITensor}(undef, N+1)
    eR = ITensor(1)
    for j in reverse(N+1:NR)
        eR *= get(ψR,j) * AR[j]
    end
    E[N+1] = eR
    for j in reverse(1:N)
        E[j]  = E[j+1] * get(ψR,j) * AR[j] * ψL[j]'
    end

    # E[1] is the full overlap <ψL|AR|ψR> before truncation 
    ov_before =  scalar(E[1])

    # Step 2: left boundary at site 1
    R        = ψR[1] * AR[1]
    L        = ψLp[1]
    l_renorm = nothing

    # Step 3: sweep left → right
    for j in 2:N
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        rho = E[j] * R * L
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = if preserve_mps_tags
            linkR = j-1 < nR ? linkind(ψR, j-1) : nothing
            isnothing(linkR) ? "" : tags(linkR)
        else
            "Link,l=$(j-1)"
        end
        tsL = if preserve_mps_tags
            linkL = j-1 < N ? linkind(ψL, j-1) : nothing
            isnothing(linkL) ? "" : tags(linkL)
        else
            "Link,l=$(j-1)"
        end

        Lis = isnothing(l_renorm) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], l_renorm)
        F   = svd(rho, Lis; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        l_renorm = F.u

        ψR_out[j-1] = U
        ψL_out[j-1] = V

        R = dag(U) * R * get(ψR, j) * AR[j] 
        L = dag(V) * L * ψLp[j]

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j-1, 1:length(Svec)] .= Svec
    end


    redge_R = ITensors.OneITensor()
    for j = reverse(N+1:NR)
        redge_R *= AR[j] * get(ψR, j)
    end
    
    ψR_out[N] = R * redge_R
    ψL_out[N] = L

    return ψL_out, ψR_out, S_all, ov_before
end
