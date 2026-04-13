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

    @assert length(AL) >= length(ψL)
    @assert length(AR) >= length(ψR)

    AL  = AL'
    ψL  = ψL''

    if direction == :right
        return _tlrcontract_rtm_right(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    elseif direction == :left
        return _tlrcontract_rtm_left(ψL, AL, AR, ψR; cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)
    else
        error("direction must be :left or :right, got :$(direction)")
    end
end


""" Builds LEFT environments, sweeps RIGHT→LEFT"""
function _tlrcontract_rtm_right(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

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
        R = R * get(ψR, j,) * AR[j]
    end

    L = get(ψL, NL) * ALp[NL]
    for j in reverse(N:NL-1)
        L = L * get(ψL, j) * ALp[j]
    end

    r_renorm = nothing

    # Step 3: sweep right → left, extracting one tensor pair per bond.
    for j in reverse(1:N-1)
        maxdim = min(dim(commoninds(R, E[j])), dim(commoninds(L, E[j])), requested_maxdim)
        rho    = E[j] * L * R

        tsR = preserve_mps_tags ? (l = linkind(ψR, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j); isnothing(l) ? "" : tags(l)) : "Link,l=$(j)"

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)

        F = svd(rho, Ris; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        r_renorm = F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = dag(U) * R * get(ψR, j) * AR[j]
        L = dag(V) * L * get(ψL, j) * ALp[j]

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    return ψL_out, ψR_out, S_all
end


""" Builds RIGHT environments, sweeps LEFT→RIGHT """
function _tlrcontract_rtm_left(ψL::MPS, AL::MPO, AR::MPO, ψR::MPS;
        cutoff, maxdim, mindim, preserve_mps_tags, kwargs...)

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
    R_l = ψR[1] * AR[1]
    L_l = ψL[1] * ALp[1]

    l_renorm = nothing

    # Step 3: sweep left → right, extracting one tensor pair per bond.
    for j in 2:n
        @show j 
        maxdim = min(dim(commoninds(R_l, E[j])), dim(commoninds(L_l, E[j])), requested_maxdim)
        rho    = R_l * L_l * E[j]

        tsR = preserve_mps_tags ? (l = linkind(ψR, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"
        tsL = preserve_mps_tags ? (l = linkind(ψL, j-1); isnothing(l) ? "" : tags(l)) : "Link,l=$(j-1)"

        # sR[j-1] is the free AR site index at the bond being extracted.
        Lis = isnothing(l_renorm) ? IndexSet(sR[j-1]) : IndexSet(sR[j-1], l_renorm)

        F = svd(rho, Lis; cutoff, maxdim, mindim, lefttags=tsR, righttags=tsL, kwargs...)
        S, U, V  = F.S, F.U, F.V
        l_renorm = F.u  # U-side bond absorbed into R_l; must appear in next Lis

        ψR_out[j-1] = U
        ψL_out[j-1] = V

        R_l = dag(U) * R_l * get(ψR, j, ITensors.OneITensor()) * get(AR, j)
        L_l = dag(V) * L_l * get(ψL, j, ITensors.OneITensor()) * get(ALp, j)

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j-1, 1:length(Svec)] .= Svec
    end

    ψR_out[n] = R_l
    ψL_out[n] = L_l


    for j = n+1:NL
        #@show inds(ψL_out[n])
        #@show inds(AL[j-1])
        ψL_out[n] *= AL[j-1]
    end

    for j = n+1:NR
        #@show inds(ψR_out[n])
        #@show inds(AR[j-1])
        ψR_out[n] *= AR[j-1]
    end
    


    #@show inds(ψR_out[1])
    #@show inds(ψR_out[end])

    #@show inds(ψL_out[1])
    #@show inds(ψL_out[end])

    return ψL_out, ψR_out, S_all
end
