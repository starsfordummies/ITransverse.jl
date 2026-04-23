### Indices prime contractions conventions
# ψL--p'--AL--p--     --p'--AR--p--ψR
# AL  (= AL'):              site plevs 1,2 — used in environments (traces over ψL'')
# ALp (= AL' with 1→3):    site plevs 2,3 — keeps plev-3 open as the output site index

function tlrcontract(::Algorithm"RTM",
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        compute_ov_before::Bool = true, # we actually do it anyway 
        direction = :right,
        kwargs...,
    )

    if direction == :right
        L, R, sv, ovb = _tlrcontract_rtm_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        L, R, sv, ovb = _tlrcontract_rtm_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end

    # We always compute ov_before, since it's cheap, but we can mask if we don't wan it 
    if !compute_ov_before
        ovb = NaN
    end
    return TruncLR(L, R, sv, ovb, NaN)  
end




""" Builds LEFT environments, sweeps RIGHT→LEFT (opt. tauB = trA(tau) )"""
function _tlrcontract_rtm_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    NL = length(AL)
    NR = length(AR)
    n = min(NL, NR)
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

    # Step 1: left environments E[j] 
    E = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env  = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "Bad env[$j] ? - $(inds(env))"
    end

    ov_before = scalar(env)

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


    return ψL_out, ψR_out, S_all, ov_before
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT (opt. tauA=trB(tau) ) """
function _tlrcontract_rtm_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    AL  = AL'
    ψL  = ψL''

    ALp = replaceprime(AL, 1 => 3, tags="Site")

    NL = length(ALp)
    NR = length(AR)
    n = min(NL, NR)
    N = max(NL, NR)

    @assert NL >= length(ψL)
    @assert NR >= length(ψR)


    sR             = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out         = typeof(ψR)(n)
    ψL_out         = typeof(ψL)(n)
    S_all          = zeros(Float64, n-1, requested_maxdim)

    # Step 1: right environments 
    E = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in reverse(1:N)                                        
        env = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "$j - $(inds(env))"
    end

    # E[1] is the full overlap <ψL|AL AR|ψR> before truncation
    ov_before = scalar(env) 

    # Build left boundary tensors at site 1.
    # ALp (site plevs 3,2): plev-2 contracts with ψL'', plev-3 stays open for output.
    R = ψR[1] * AR[1]
    L = ψL[1] * ALp[1]

    renorm_idx = nothing

    # Step 3: sweep left → right
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

        Svec = collect(storage(S).data) ./ sum(S)
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
    
    ψR_out[n] = R * redge_R
    ψL_out[n] = L * redge_L

    return ψL_out, ψR_out, S_all, ov_before
end
