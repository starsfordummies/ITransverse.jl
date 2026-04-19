### Indices prime contractions conventions
# ψL--p'--AL--p--     --p'--AR--p--ψR
# AL  (= AL'):              site plevs 1,2 — used in environments (traces over ψL'')
# ALp (= AL' with 1→3):    site plevs 2,3 — keeps plev-3 open as the output site index

Base.get(psi::AbstractMPS, j::Integer, default=ITensor(1)) = 
    1 <= j <= length(psi) ? psi[j] : default

function tlrcontract(::Algorithm"RTM",
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        direction = :right,
        compute_norm::Bool = false,
        kwargs...,
    )

    if direction == :right
        L, R, sv, nf = _tlrcontract_rtm_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)
    elseif direction == :left
        L, R, sv, nf = _tlrcontract_rtm_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
    return TruncLR(L, R, sv, nf)
end




""" Builds LEFT environments, sweeps RIGHT→LEFT (opt. tauB = trA(tau) )"""
function _tlrcontract_rtm_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)

    NL = length(AL)
    NR = length(AR)
    n  = min(NL, NR)
    N = max(NL, NR)

    @assert NL >= length(ψL)
    @assert NR >= length(ψR)

    AL  = AL'
    ψL  = ψL''
    ALp = replaceprime(AL, 1 => 3, tags="Site")

    sR             = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out         = typeof(ψR)(n)
    ψL_out         = typeof(ψL)(n)
    S_all          = zeros(Float64, n-1, requested_maxdim)

    # Step 1: left environments E[j] = contraction of sites 1..j (both arms).
    E = Vector{ITensor}(undef, N)
    for j in 1:N
        prev  = j == 1 ? ITensors.OneITensor() : E[j-1]
        E[j]  = prev * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        @assert ndims(E[j]) < 5 "Bad env[$j] ? - $(inds(E[j]))"
    end

    ov_before = compute_norm ? scalar(E[N]) : 1.0

    # Initialize right boundary tensors 
    R = get(ψR, NR) * AR[NR]
    for j in reverse(n:NR-1)
        R = R * get(ψR, j) * AR[j]
    end

    L = get(ψL, NL) * ALp[NL]
    for j in reverse(n:NL-1)
        L = L * get(ψL, j) * ALp[j]
    end

    renorm_idx = nothing

    # Step 3: sweep right → left
    for j in reverse(1:n-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)
        rho    = E[j] * L * R

        tsR = preserve_mps_tags ? (l = linkind(ψR, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], renorm_idx)

        # We associate the "U" SVD branch to the R, the "V" to the L 
        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        renorm_idx = F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = dag(U) * R * get(ψR, j) * AR[j]
        L = dag(V) * L * get(ψL, j) * ALp[j]

        #@show ndims(R), ndims(L)

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    norm_factor = if compute_norm 
        ov_after = overlap_noconj(noprime(ψL_out), noprime(ψR_out))
        ov_before / ov_after
    else
        1.0
    end

    return ψL_out, ψR_out, S_all, norm_factor
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT (opt. tauA=trB(tau) ) """
function _tlrcontract_rtm_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)

    @assert length(AL) >= length(ψL)
    @assert length(AR) >= length(ψR)

    AL  = AL'
    ψL  = ψL''

    ALp = replaceprime(AL, 1 => 3, tags="Site")

    NL = length(ALp)
    NR = length(AR)
    n  = min(NL, NR)
    N = max(NL, NR, length(ψL), length(ψR))

    sR             = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out         = typeof(ψR)(n)
    ψL_out         = typeof(ψL)(n)
    S_all          = zeros(Float64, n-1, requested_maxdim)

    # Step 1: right environments E[j] = contraction of sites j..N (+ any excess beyond N).
    # E[N+1] is initialised to OneITensor() and accumulates any excess sites first.
    E = Vector{ITensor}(undef, N+1)
    E[N+1] = ITensor(1.)
    for j in N:-1:1                                        
        E[j] = E[j+1] * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        @assert ndims(E[j]) < 5 "$j - $(inds(E[j]))"
    end

    # E[1] is the full overlap <ψL|AL AR|ψR> before truncation (scalar for OBC)
    ov_before = compute_norm ? scalar(E[1]) : 1.0

    # Step 2: left boundary tensors at site 1.
    # ALp (site plevs 2,3): plev-2 contracts with ψL'', plev-3 stays open for output.
    R = ψR[1] * AR[1]
    L = ψL[1] * ALp[1]

    renorm_idx = nothing

    # Step 3: sweep left → right, extracting one tensor pair per bond.
    for j in 2:n
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)
        rho    = E[j] * R * L 

        tsR = preserve_mps_tags ? (l = linkind(ψR, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"

        Ris = isnothing(renorm_idx) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], renorm_idx)

        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        renorm_idx = F.u  

        ψR_out[j-1] = U
        ψL_out[j-1] = V

        R = dag(U) * R * get(ψR, j) * get(AR, j)
        L = dag(V) * L * get(ψL, j) * get(ALp, j)

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j-1, 1:length(Svec)] .= Svec
    end

    redge_L = ITensors.OneITensor()
    for j = reverse(n+1:NL)
        redge_L *= AL[j] * get(ψL, j)
    end

    redge_R = ITensors.OneITensor()
    for j = reverse(n+1:NR)
        redge_R *= AR[j] * get(ψR, j)
    end
    
    #@show inds(R)
    #@show inds(redge_R)

    ψR_out[n] = R * redge_R
    ψL_out[n] = L * redge_L

    norm_factor = if compute_norm 
        ov_after = overlap_noconj(noprime(ψL_out), noprime(ψR_out))
        ov_before / ov_after
    else
        1.0
    end

    return ψL_out, ψR_out, S_all, norm_factor
end



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
        direction = :right,
        compute_norm::Bool = false,
        kwargs...,
    )
    if direction == :right
        L, R, sv, nf = _trcontract_rtm_right(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)
    elseif direction == :left
        L, R, sv, nf = _trcontract_rtm_left(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
    return TruncLR(L, R, sv, nf)
end

""" Builds LEFT environments, sweeps RIGHT→LEFT """
function _trcontract_rtm_right(ψL::MPS, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)

    ### Indices prime contractions conventions
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    N  = length(ψL)
    nR = length(ψR)
    NR = length(AR)

    #@info N, nR, NR

    @assert NR >= nR

    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = typeof(ψR)(N)
    ψL_out           = typeof(ψL)(N)
    ψLp              = prime(siteinds, ψL')
    S_all            = zeros(Float64, N-1, requested_maxdim)

    # Step 1: build left environments E[j] = sites 1..j
    E   = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env   = env * get(ψR,j) * AR[j] * ψL[j]' 
        E[j]  = env
        @assert ndims(env) < 4
    end

    # E[N] is the full overlap <ψL|AR|ψR> before truncation (scalar for OBC)
    ov_before = compute_norm ? scalar(E[N]) : 1.0

    # Step 2: right boundary, absorbing excess AR sites beyond N
    R = ITensors.OneITensor()
    for j in reverse(N:NR)
        R = R * get(ψR,j) * AR[j] 
    end
    L = ψLp[N]

    r_renorm = nothing

    # Step 3: sweep right → left
    for j in reverse(1:N-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)

        rho = E[j] * L * R
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = if preserve_mps_tags
            linkR = j < nR ? linkind(ψR, j) : nothing
            isnothing(linkR) ? "" : tags(linkR)
        else
            "Link,l=$(j)"
        end
        tsL = if preserve_mps_tags
            linkL = j < N ? linkind(ψL, j) : nothing
            isnothing(linkL) ? "" : tags(linkL)
        else
            "Link,l=$(j)"
        end

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)
        F   = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        r_renorm = F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = dag(U) * R * get(ψR,j) * AR[j] 
        L = dag(V) * L * ψLp[j]

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    norm_factor = if compute_norm && ov_before !== nothing
        ov_after = overlap_noconj(noprime(ψL_out), noprime(ψR_out))
        ov_before / ov_after
    else
        1.0
    end

    return ψL_out, ψR_out, S_all, norm_factor
end

""" Builds RIGHT environments, sweeps LEFT→RIGHT """
function _trcontract_rtm_left(ψL::MPS, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, compute_norm, kwargs...)

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

    # E[1] is the full overlap <ψL|AR|ψR> before truncation (scalar for OBC)
    ov_before = compute_norm ? scalar(E[1]) : 1.0

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

    norm_factor = if compute_norm && ov_before !== nothing
        ov_after = overlap_noconj(noprime(ψL_out), noprime(ψR_out))
        ov_before / ov_after
    else
        1.0
    end

    return ψL_out, ψR_out, S_all, norm_factor
end
