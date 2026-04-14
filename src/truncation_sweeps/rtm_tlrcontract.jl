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
        kwargs...,
    )

    if direction == :right
        return _tlrcontract_rtm_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        return _tlrcontract_rtm_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
end




""" Builds LEFT environments, sweeps RIGHT→LEFT"""
function _tlrcontract_rtm_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)


    @assert length(AL) >= length(ψL)
    @assert length(AR) >= length(ψR)

    AL  = AL'
    ψL  = ψL''

    ALp = replaceprime(AL, 1 => 3, tags="Site")

    NL = length(ALp)
    NR = length(AR)
    N  = min(NL, NR)

    sR             = firstsiteinds(AR, plev=1)
    requested_maxdim = maxdim
    ψR_out         = typeof(ψR)(N)
    ψL_out         = typeof(ψL)(N)
    S_all          = zeros(Float64, N-1, requested_maxdim)

    # Step 1: left environments E[j] = contraction of sites 1..j (both arms).
    # get(ψ, j, OneITensor()) returns OneITensor() for j beyond the MPS, so no branching needed.
    E = Vector{ITensor}(undef, N-1)
    for j in 1:N-1
        prev  = j == 1 ? ITensors.OneITensor() : E[j-1]
        E[j]  = prev * get(ψR, j) * AR[j] * AL[j] * get(ψL, j)
        @assert ndims(E[j]) < 5 "Bad env[$j] ? - $(inds(E[j]))"
    end


    # Step 2: right boundary tensors (AR arm = R, ALp arm = L).
    # When NR==N the loop body is reverse(N:N-1) = empty, so R is just site N.
    R = get(ψR, NR) * AR[NR]
    for j in reverse(N:NR-1)
        R = R * get(ψR, j) * AR[j]
    end

    L = get(ψL, NL) * ALp[NL]
    for j in reverse(N:NL-1)
        L = L * get(ψL, j) * ALp[j]
    end

    renorm_idx = nothing

    # Step 3: sweep right → left, extracting one tensor pair per bond.
    for j in reverse(1:N-1)
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

    return ψL_out, ψR_out, S_all
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT """
function _tlrcontract_rtm_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

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


    return ψL_out, ψR_out, S_all
end








function trcontract(::Algorithm"RTM",
        ψL::MPS,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        direction = :right, # TODO no left yet 
        kwargs...,
    )

    @assert direction == :right ":left not implemented yet"

    ### Indices prime contractions conventions
    # (ie we prime ψL to contract it to AL in the proper direction)
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    N = length(ψL)
    nR = length(ψR)
    NR = length(AR)

    @assert NR >= nR

    sR  = firstsiteinds(AR, plev=1)

    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    # Step 1: build left environments up to site N-1
    E = Vector{ITensor}(undef, N-1)

    # Sites 1..min(nL,nR): both ψL and ψR present
    # Sites min+1..max(nL,nR): only one ψ present
    # Sites max+1..N-1: neither ψ present
    env = ITensors.OneITensor() 
    for j in 1:N-1
        hasψR = j <= nR
        env = if hasψR 
            env * ψR[j] * AR[j]  * ψL[j]'
        else
            env * AR[j]  * ψL[j]'
        end
        E[j] = env

        @assert ndims(env) < 4
    end

    # Step 2: initialize R and L by contracting the excess tail of the longer MPO
    # Start from the far right edge and work inward until both sides reach site N

    # Initialize edge tensors at the right boundary
    R = if NR > N
        # contract ψR and AR from NR down to N+1
        t = nR >= NR ? ψR[NR] * AR[NR] : AR[NR]
        for j in reverse(N:NR-1)
            t = t * (j <= nR ? ψR[j] * AR[j] : AR[j])
        end
        t
    else
        # NR == N, initialize at site N only
        nR >= NR ? ψR[NR] * AR[NR] : AR[NR]
    end

    ψLp = prime(siteinds, ψL')

    L = ψLp[N]


    r_renorm = nothing
    S_all = zeros(Float64, N-1, requested_maxdim)

    # Step 3: main sweep from N-1 down to 1
    for j in reverse(1:N-1)

        hasψR = j <= nR

        ciR = commoninds(R, E[j])
        ciL = commoninds(L, E[j])
        maxdim = min(dim(ciR), dim(ciL), requested_maxdim)

        #@show inds(E[j])
        #@show inds(L)
        #@show inds(R)
        rho = E[j] * L * R
        #@show inds(rho)
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = if preserve_mps_tags
            linkR  = j < nR ? linkind(ψR, j) : nothing
            tsR = isnothing(linkR) ? "" : tags(linkR)
        else
            "Link,l=$(j)"
        end

        tsL = if preserve_mps_tags
            linkL  = j < N ? linkind(ψL, j) : nothing
            tsL = isnothing(linkL) ? "" : tags(linkL)
        else
            "Link,l=$(j)"
        end

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1])  : IndexSet(sR[j+1], r_renorm)

        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm= F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = hasψR ? dag(U) * R * ψR[j] * AR[j]  : dag(U) * R * AR[j]
        L = dag(V) * L * ψLp[j] 

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    return ψL_out, ψR_out, S_all
end
