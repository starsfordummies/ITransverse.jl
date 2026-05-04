
tocpu(x) = adapt(Array, x)

""" Contract MPO-MPS with algorithm densitymatrix, starting from the left. At the end we can chop/extend 
if we work with light cone. If `contract_dangling=true` we shrink dangling edges, if present  """
function ITransverse.tcontract(::Algorithm{:cudensitymatrix},
        A::MPO,
        ψ::MPS;
        cutoff = 1.0e-13,
        maxdim = maxlinkdim(A) * maxlinkdim(ψ),
        mindim = 1,
        direction = :right,  # TODO not implemented yet 
        contract_dangling::Bool=true,
        kwargs...,
    )

    @assert length(A) >= length(ψ)  "Error: length(MPO)=$(length(A)) < $(length(ψ)) = length(ψ) "

    N = length(A)
    n = length(ψ)

    mindim = max(mindim, 1)
    requested_maxdim = maxdim
    ψ_out = typeof(ψ)(N)

    # In case A and ψ have the same link indices
    A = sim(linkinds, A)

    ψ_c = dag(ψ)''
    simA_c = prime(dag(A), 2)
    A_c = replaceprime(simA_c, 3 => 1)

    # Store the right environment tensors
    E = Vector{ITensor}(undef, N)

    Ecurr = N > n ?  A[N] *A_c[N] : ψ[N] * A[N] * A_c[N] * ψ_c[N]
    E[N] = Ecurr
    for j in reverse(n+1:N-1)
        Ecurr = E[j + 1] * A[j] * A_c[j] 
        E[j] = Ecurr
    end
    # Do the bulk on GPU
    Ecurr = togpu(Ecurr)
    for j in reverse(2:min(N-1,n))
        Ecurr = Ecurr * togpu(ψ[j]) * togpu(A[j]) * togpu(A_c[j]) * togpu(ψ_c[j])
        E[j] = tocpu(Ecurr)
    end

    # @show E


    L = togpu(ψ[1] * A[1])
    simL_c = togpu(ψ_c[1] * simA_c[1])
    l_renorm = nothing
    r_renorm = nothing

    S_all = zeros(Float64, n-1, maxdim)

    for j in 1:min(n-1,N-1)

        # @show j 

        # Determine smallest maxdim to use
        cip = commoninds(ψ[j], E[j + 1])
        ciA = commoninds(A[j], E[j + 1])
        prod_dims = dim(cip) * dim(ciA)
        maxdim = min(prod_dims, requested_maxdim)

        s = siteinds(uniqueinds, A, ψ, j)
        s̃ = siteinds(uniqueinds, simA_c, ψ_c, j)
        rho = togpu(E[j + 1]) * L * simL_c
        l = linkind(ψ, j)
        ts = isnothing(l) ? "" : tags(l)
        Lis = isnothing(l_renorm) ? IndexSet(s...) : IndexSet(s..., l_renorm)
        Ris = isnothing(r_renorm) ? IndexSet(s̃...) : IndexSet(s̃..., r_renorm)

        @assert ndims(rho) < 5 " $j $(inds(rho))"

        # @show inds(rho)
        # @show Lis

        F = eigen(rho, Lis, Ris; ishermitian = true, tags = ts, cutoff, maxdim, mindim, kwargs...)
        D, U, Ut = F.D, F.V, F.Vt
        l_renorm, r_renorm = F.l, F.r

        # @show inds(U)
        # @show inds(Ut)
        # @show l_renorm, r_renorm

        ψ_out[j] = tocpu(Ut)

        L = L * dag(Ut) * togpu(ψ[j+1]) * togpu(A[j+1])
        simL_c = simL_c * U* togpu(ψ_c[j+1]) * togpu(simA_c[j+1])

        Dvec = collect(D.tensor.storage.data)/sum(D)  
 
        S_all[j, 1:length(Dvec)] .= Dvec  
        #@show sum(Dvec)
    
    end

    ψ_out[n] = tocpu(L)


    for j = n+1:N 
        ψ_out[j] = A[j]
    end


    #@info "Setting ortho lims $(n-1):$(N+1)"
    ITensorMPS.setleftlim!(ψ_out, n-1)
    ITensorMPS.setrightlim!(ψ_out, N+1)

    if contract_dangling
        ITransverse.contract_dangling!(ψ_out)
    end

end

# ─── cuRTM trcontract ───────────────────────────────────────────────────────────────────

""" RTM truncation with GPU-accelerated inner sweep (mirrors :RTM but loads
tensors to GPU on-the-fly, keeping environments on CPU). """
function ITransverse.trcontract(::Algorithm"cuRTM",
        ψL::MPS,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        compute_ov_before::Bool = true,
        direction = :right,
        kwargs...,
    )
    if direction == :right
        L, R, sv, ovb = _cutrcontract_rtm_right(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        L, R, sv, ovb = _cutrcontract_rtm_left(ψL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end

    if !compute_ov_before
        ovb = NaN
    end

    return ITransverse.TruncLR(L, R, sv, ovb, NaN)
end

function _cutrcontract_rtm_right(ψL::MPS, AR::MPO, ψR::MPS;
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

    # Step 1: build right environments on CPU
    E = Vector{ITensor}(undef, N+1)
    eR = ITensor(1)
    for j in reverse(N+1:NR)
        eR *= get(ψR,j) * AR[j]
    end
    E[N+1] = eR
    for j in reverse(1:N)
        E[j] = E[j+1] * get(ψR,j) * AR[j] * ψL[j]'
    end

    ov_before = scalar(E[1])

    # Step 2: left boundary — move running tensors to GPU
    R        = togpu(ψR[1] * AR[1])
    L        = togpu(ψLp[1])
    l_renorm = nothing

    # Step 3: sweep left → right on GPU
    for j in 2:N
        maxdim = min(dim(commoninds(tocpu(R), E[j])), dim(commoninds(tocpu(L), E[j])), requested_maxdim)

        rho = togpu(E[j]) * R * L
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

        ψR_out[j-1] = tocpu(U)
        ψL_out[j-1] = tocpu(V)

        R = dag(U) * R * togpu(get(ψR, j)) * togpu(AR[j])
        L = dag(V) * L * togpu(ψLp[j])

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j-1, 1:length(Svec)] .= Svec
    end

    redge_R = ITensor(1)
    for j in reverse(N+1:NR)
        redge_R *= AR[j] * get(ψR, j)
    end

    ψR_out[N] = tocpu(R) * redge_R
    ψL_out[N] = tocpu(L)

    return ψL_out, ψR_out, S_all, ov_before
end

function _cutrcontract_rtm_left(ψL::MPS, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    nL = length(ψL)
    nR = length(ψR)
    N  = length(AR)

    @assert N >= nR
    @assert N >= nL

    ψLp              = ψL'
    ψLpp             = prime(siteinds, ψLp)
    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = MPS(nL)
    ψL_out           = MPS(nL)
    S_all            = zeros(Float64, N-1, requested_maxdim)

    # Step 1: build left environments on CPU
    E   = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env  = env * get(ψR,j) * AR[j] * get(ψLp,j)
        E[j] = env
        @assert ndims(env) < 4 "env[$(j)] ? $(inds(env))"
    end

    ov_before = scalar(E[N])

    # Step 2: right boundary — move running tensors to GPU
    R = ITensor(1)
    for j in reverse(nL:N)
        R *= get(ψR,j) * AR[j]
    end
    R        = togpu(R)
    L        = togpu(ψLpp[nL])
    r_renorm = nothing

    # Step 3: sweep right → left on GPU
    for j in reverse(1:nL-1)
        maxdim = min(dim(commoninds(tocpu(R), E[j])), dim(commoninds(tocpu(L), E[j])), requested_maxdim)

        rho = togpu(E[j]) * L * R
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = preserve_mps_tags ? tags(linkind(ψR, j)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? tags(linkind(ψL, j)) : "Link,l=$(j)"

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)
        F   = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        r_renorm = F.u

        ψR_out[j+1] = tocpu(U)
        ψL_out[j+1] = tocpu(V)

        R = dag(U) * R * togpu(get(ψR,j)) * togpu(AR[j])
        L = dag(V) * L * togpu(get(ψLpp,j))

        Svec = collect(storage(S).data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = tocpu(R)
    ψL_out[1] = tocpu(L)

    return ψL_out, ψR_out, S_all, ov_before
end

# ─── cuRTM tlrcontract ────────────────────────────────────────────────────

""" Two-MPO RTM truncation with GPU-accelerated inner sweep. """
function ITransverse.tlrcontract(::Algorithm"cuRTM",
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        compute_ov_before::Bool = true,
        direction = :right,
        kwargs...,
    )
    if direction == :right
        L, R, sv, ovb = _cutlrcontract_rtm_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        L, R, sv, ovb = _cutlrcontract_rtm_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
    if !compute_ov_before
        ovb = NaN
    end
    return ITransverse.TruncLR(L, R, sv, ovb, NaN)
end

function _cutlrcontract_rtm_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    NL = length(AL)
    NR = length(AR)
    n  = min(NL, NR)
    N  = max(NL, NR)

    @assert NL >= length(ψL)
    @assert NR >= length(ψR)

    AL  = AL'
    ψL  = ψL''
    ALp = replaceprime(AL, 1 => 3, tags="Site")

    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = typeof(ψR)(n)
    ψL_out           = typeof(ψL)(n)
    S_all            = zeros(Float64, n-1, requested_maxdim)

    # Step 1: build left environments on CPU
    E   = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in 1:N
        env  = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "Bad env[$j] ? - $(inds(env))"
    end
    ov_before = scalar(env)

    # Step 2: right boundary — move to GPU
    R = get(ψR, NR) * AR[NR]
    for j in reverse(n:NR-1)
        R = R * get(ψR, j) * AR[j]
    end
    L = get(ψL, NL) * ALp[NL]
    for j in reverse(n:NL-1)
        L = L * get(ψL, j) * ALp[j]
    end
    R          = togpu(R)
    L          = togpu(L)
    renorm_idx = nothing

    # Step 3: sweep right → left on GPU
    for j in reverse(1:n-1)
        maxdim = min(dim(commoninds(tocpu(R), E[j])), dim(commoninds(tocpu(L), E[j])), requested_maxdim)
        rho    = togpu(E[j]) * L * R

        tsR = preserve_mps_tags ? (l = linkind(ψR, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        Ris = isnothing(renorm_idx) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], renorm_idx)

        F  = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V = F.S, F.U, F.V
        renorm_idx = F.u

        ψR_out[j+1] = tocpu(U)
        ψL_out[j+1] = tocpu(V)

        R = dag(U) * R * togpu(get(ψR, j)) * togpu(AR[j])
        L = dag(V) * L * togpu(get(ψL, j)) * togpu(ALp[j])

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = tocpu(R)
    ψL_out[1] = tocpu(L)

    return ψL_out, ψR_out, S_all, ov_before
end

function _cutlrcontract_rtm_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

    AL  = AL'
    ψL  = ψL''
    ALp = replaceprime(AL, 1 => 3, tags="Site")

    NL = length(ALp)
    NR = length(AR)
    n  = min(NL, NR)
    N  = max(NL, NR)

    @assert NL >= length(ψL)
    @assert NR >= length(ψR)

    sR               = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out           = typeof(ψR)(n)
    ψL_out           = typeof(ψL)(n)
    S_all            = zeros(Float64, n-1, requested_maxdim)

    # Step 1: build right environments on CPU
    E   = Vector{ITensor}(undef, N)
    env = ITensors.OneITensor()
    for j in reverse(1:N)
        env  = env * get(ψR, j) * get(AR, j) * get(AL, j) * get(ψL, j)
        E[j] = env
        @assert ndims(env) < 5 "$j - $(inds(env))"
    end
    ov_before = scalar(env)

    # Step 2: left boundary — move to GPU
    R          = togpu(ψR[1] * AR[1])
    L          = togpu(ψL[1] * ALp[1])
    renorm_idx = nothing

    # Step 3: sweep left → right on GPU
    for j in 2:n
        maxdim = min(dim(commoninds(tocpu(R), E[j])), dim(commoninds(tocpu(L), E[j])), requested_maxdim)
        rho    = togpu(E[j]) * R * L

        tsR = preserve_mps_tags ? (l = linkind(ψR, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        Lis = isnothing(renorm_idx) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], renorm_idx)

        F  = svd(rho, Lis; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V = F.S, F.U, F.V
        renorm_idx = F.u

        ψR_out[j-1] = tocpu(U)
        ψL_out[j-1] = tocpu(V)

        R = dag(U) * R * togpu(get(ψR, j)) * togpu(get(AR, j))
        L = dag(V) * L * togpu(get(ψL, j)) * togpu(get(ALp, j))

        Svec = collect(storage(S).data) ./ sum(S)
        S_all[j-1, 1:length(Svec)] .= Svec
    end

    redge_L = ITensors.OneITensor()
    for j in reverse(n+1:NL)
        redge_L *= AL[j] * get(ψL, j)
    end
    redge_R = ITensors.OneITensor()
    for j in reverse(n+1:NR)
        redge_R *= AR[j] * get(ψR, j)
    end

    ψR_out[n] = tocpu(R) * redge_R
    ψL_out[n] = tocpu(L) * redge_L

    return ψL_out, ψR_out, S_all, ov_before
end
