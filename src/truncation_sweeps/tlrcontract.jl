""" 
Applies MPO to left-right and truncates on the RTM |AR R><L AL| \\
Allows for different length MPS/MPO \\
Returns LEFT, RIGHT, SV
"""
function tlrcontract(
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        direction = :right, # TODO no left yet 
        kwargs...,
    )

    @assert direction == :right ":left not implemented yet"


    ### Indices prime contractions conventions
    # (ie we prime ψL to contract it to AL in the proper direction)
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    nL = length(ψL)
    nR = length(ψR)
    NL = length(AL)
    NR = length(AR)
    N  = min(NL, NR)  # output MPS length

    @assert NL >= nL
    @assert NR >= nR

    sR  = firstsiteinds(AR, plev=1)

    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    AL  = swapprime(AL, 0 => 1, tags="Site")
    ALp = replaceprime(AL, 1 => 2)

    # Step 1: build left environments up to site N-1
    E = Vector{ITensor}(undef, N-1)

    # Sites 1..min(nL,nR): both ψL and ψR present
    # Sites min+1..max(nL,nR): only one ψ present
    # Sites max+1..N-1: neither ψ present
    for j in 1:N-1
        prev = j == 1 ? ITensors.OneITensor() : E[j-1]
        hasψR = j <= nR
        hasψL = j <= nL
        E[j] = if hasψR && hasψL
            prev * ψR[j] * AR[j] * AL[j] * ψL[j]
        elseif hasψR
            prev * ψR[j] * AR[j] * AL[j]
        elseif hasψL
            prev * AR[j] * AL[j] * ψL[j]
        else
            prev * AR[j] * AL[j]
        end
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

    L = if NL > N
        t = nL >= NL ? ψL[NL] * ALp[NL] : ALp[NL]
        for j in reverse(N:NL-1)
            t = t * (j <= nL ? ψL[j] * ALp[j] : ALp[j])
        end
        t
    else
        nL >= NL ? ψL[NL] * ALp[NL] : ALp[NL]
    end

    r_renorm = nothing
    S_all = zeros(Float64, N-1, requested_maxdim)

    # Step 3: main sweep from N-1 down to 1
    for j in reverse(1:N-1)

        hasψR = j <= nR
        hasψL = j <= nL

        ciR = commoninds(R, E[j])
        ciL = commoninds(L, E[j])
        maxdim = min(dim(ciR), dim(ciL), requested_maxdim)

        rho = E[j] * L * R
        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        tsR = if preserve_mps_tags
            linkR  = j < nR ? linkind(ψR, j) : nothing
            tsR = isnothing(linkR) ? "" : tags(linkR)
        else
            "Link,l=$(j)"
        end

        tsL = if preserve_mps_tags
            linkL  = j < nL ? linkind(ψL, j) : nothing
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
        L = hasψL ? dag(V) * L * ψL[j] * ALp[j] : dag(V) * L * ALp[j]

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    return ψL_out, ψR_out, S_all
end


#### NEW LR apply 
function tlrapply(::Algorithm"RTM", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    tpsiL, tpsiR, sv = tlrcontract(psiL, OL, OR, psiR; kwargs...)
    return replaceprime(tpsiL,  2 => 0), replaceprime(tpsiR,  1 => 0), sv
end



function tlrapply(::Algorithm"naiveRTM", psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
            OpsiR = applyn(OR, psiR; truncate=false)
            psiLO = applyns(OL, psiL; truncate=false)  
            truncate_sweep(psiLO, OpsiR; kwargs...)
end





### Apply only one, then truncate 



""" 
Applies MPO to the right and truncates on the RTM |AR R><L| by building it explicitly \\
Returns LEFT, RIGHT, SV
"""
function trcontract(
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

        @show inds(E[j])
        @show inds(L)
        @show inds(R)
        rho = E[j] * L * R
        @show inds(rho)
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
