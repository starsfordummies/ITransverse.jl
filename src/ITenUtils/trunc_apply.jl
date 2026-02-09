using ITensors.TagSets: commontags

struct TruncatedMPS{T<:Number}
    psi::MPS 
    SV::Matrix{T}
end

TruncatedMPS(psi::MPS; SV=zeros(Float64, 1,1)) = TruncatedMPS(psi, SV)


""" Custom truncate! that returns SV spectra at each bipartition as a matrix (Nlinks x maxdim) """
function ttruncate!(
        M::AbstractMPS;
        site_range = 1:length(M),
        callback = Returns(nothing), 
        maxdim=maxlinkdim(M),
        kwargs...
    )
    # Left-orthogonalize all tensors to make
    # truncations controlled
    orthogonalize!(M, last(site_range))

    # Pre-allocate matrix for singular values
    # Rows: bipartitions (N-1 of them)
    # Cols: singular values (up to maxdim)
    SV_matrix = zeros(Float64, length(M)-1, maxdim)

    # Perform truncations in a right-to-left sweep
    for j in reverse((first(site_range) + 1):last(site_range))
        rinds = uniqueinds(M[j], M[j - 1])
        ltags = tags(commonind(M[j], M[j - 1]))
        U, S, V, spec = svd(M[j], rinds; lefttags = ltags, kwargs...)
        M[j] = U
        M[j - 1] *= (S * V)
        setrightlim!(M, j)
        callback(; link = (j => j - 1), truncation_error = spec.truncerr)

        s_vec = Array(storage(S).data)
        n_s = min(length(s_vec), maxdim)
            
        # Safe CPU operation
        @views SV_matrix[j-1, 1:n_s] .= s_vec[1:n_s]

    end
    return M, SV_matrix
end




""" Applies MPO to MPS from the other side (p' MPO leg, leaving open the p leg): 
instead of 

``` psi--(p)-(p)--[O]--(p')-- =  Opsi--(p)-- ```

it contracts  

``` --(p)--[O]--(p')-(p)--psi =  -(p)--Opsi ```
"""
function tapplys(alg, O::MPO, psi::AbstractMPS; kwargs...)
    tpsi, sv = tcontract(alg, O, prime(siteinds,psi); kwargs...)
    return replaceprime(tpsi, 2 => 0), sv
end

tapply(a::AbstractMPS,b::AbstractMPS; alg="naive", kwargs...) = tapply(Algorithm(alg), a,b; kwargs...)

function tapply(alg, O::MPO, psi::MPS; kwargs...)
    tpsi, sv = tcontract(alg, O, psi; kwargs...)
    return replaceprime(tpsi,  1 => 0), sv
end

function tapply(alg, O::MPO, Q::MPO; kwargs...)
    tpsi, sv = tcontract(alg, O, sim(linkinds, Q); kwargs...)
    return replaceprime(tpsi,  1 => 0), sv
end

# TODO Check: For MPOs, applyns(A,B) = apply(Algorithm"naive",B,A) ? 


""" Copied from ITensorMPS's `contract` but adapted so that it can also extend. 
Extension is done at the right (top), ie. contraction goes like 
```
    | | | | | | |
A = o-o-o-o-o-o-o 
    | | | | |  
psi o o o o o      
```
"""
function tcontract(::Algorithm"naive", A::MPO, ψ::AbstractMPS; preserve_tags_mps::Bool=false, truncate=false, kwargs...)

    # TODO Add offset for contraction to allow extension on both edges 
    @assert length(A) >= length(ψ)

    N = length(A)
    n = min(length(A), length(ψ))

    ψ_out = typeof(ψ)(N)
    for j in 1:n
        ψ_out[j] = A[j] * ψ[j]
    end
    for j = n+1:N
        ψ_out[j] = A[j] 
    end

    for b in 1:(N - 1)
        Al = linkinds(A, b)  #commoninds(A[b], A[b + 1])
        ψl = linkinds(ψ, b) #   commoninds(ψ[b], ψ[b + 1])
        l = [Al..., ψl...]
        if !isempty(l)
            ttag = preserve_tags_mps ? tags(linkind(ψ, b)) : "CMB,Link,l=$(b)"
            C = combiner(l, tags=ttag)
            ψ_out[b] *= C
            ψ_out[b + 1] *= dag(C)
        end
    end

    contract_dangling!(ψ_out)

    # If :truncp, :cutoff or :maxdim keywords are present, set truncate=true
    if haskey(kwargs, :truncp)
        (;cutoff, maxbondim) = kwargs[:truncp]
        kwargs = (;kwargs..., cutoff=cutoff, maxdim=maxbondim)
        truncate = true
    elseif haskey(kwargs, :cutoff) 
        cutoff = kwargs[:cutoff]
        maxdim = maxlinkdim(ψ_out)
        truncate = true
    elseif haskey(kwargs, :maxdim)
        maxdim = kwargs[:maxdim]
        cutoff=1e-14
        truncate = true
    end

    ψ_out, SVs = if truncate
        #@info "truncating, $(cutoff) - $(maxdim)"
        ttruncate!(ψ_out; cutoff, maxdim)
    else
        ψ_out, zeros(length(ψ_out)-1, maxlinkdim(ψ_out))
    end

    return ψ_out, SVs
end



""" Contract MPO-MPS with algorithm densitymatrix, starting from the left. At the end we can chop/extend 
if we work with light cone """
function tcontract(::Algorithm"densitymatrix",
        A::MPO,
        ψ::MPS;
        cutoff = 1.0e-13,
        maxdim = maxlinkdim(A) * maxlinkdim(ψ),
        mindim = 1,
        kwargs...,
    )

    @assert length(A) >= length(ψ)

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

    E[N] = N > n ?  A[N] * A_c[N] : ψ[N] * A[N] * A_c[N] * ψ_c[N]

    for j in reverse(n+1:N-1)
        E[j] = E[j + 1] * A[j] * A_c[j] 
    end
    for j in reverse(2:min(N-1,n))
        E[j] = E[j + 1] * ψ[j] * A[j] * A_c[j] * ψ_c[j]

    end

    # @show E


    L = ψ[1] * A[1]
    simL_c =  ψ_c[1] * simA_c[1]
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
        rho = E[j + 1] * L * simL_c
        l = linkind(ψ, j)
        ts = isnothing(l) ? "" : tags(l)
        Lis = isnothing(l_renorm) ? IndexSet(s...) : IndexSet(s..., l_renorm)
        Ris = isnothing(r_renorm) ? IndexSet(s̃...) : IndexSet(s̃..., r_renorm)

        @assert ndims(rho) < 5 " $j $(inds(rho))"

        F = eigen(rho, Lis, Ris; ishermitian = true, tags = ts, cutoff, maxdim, mindim, kwargs...)
        D, U, Ut = F.D, F.V, F.Vt
        l_renorm, r_renorm = F.l, F.r


        # F = symm_svd(rho, Lis; cutoff, maxdim, kwargs...)

        # D, U, Ut = F.S, F.U, F.U
        # l_renorm, r_renorm = F.u, F.v

        ψ_out[j] = Ut

        L = L * dag(Ut) * ψ[j+1] * A[j+1]
        simL_c = simL_c * U* ψ_c[j+1] * simA_c[j+1]

        Dvec = collect(D.tensor.storage.data)/tr(D)  
 
        S_all[j, 1:length(Dvec)] .= Dvec  
    
    end

    ψ_out[n] = L


    for j = n+1:N 
        ψ_out[j] = A[j]
    end


    #@info "Setting ortho lims $(n-1):$(N+1)"
    ITensorMPS.setleftlim!(ψ_out, n-1)
    ITensorMPS.setrightlim!(ψ_out, N+1)

    return ψ_out, S_all
end





""" Contract MPO-MPS with zipup algorithm """
function tcontract(::Algorithm"zipup",
        A::MPO,
        B::AbstractMPS;
        cutoff = 1.0e-13,
        maxdim = maxlinkdim(A) * maxlinkdim(B),
        mindim = 1,
        kwargs...,
    )
     if hassameinds(siteinds, A, B)
    error(
      "In `contract(A::MPO, B::MPO)`, MPOs A and B have the same site indices. The indices of the MPOs in the contraction are taken literally, and therefore they should only share one site index per site so the contraction results in an MPO. You may want to use `replaceprime(contract(A', B), 2 => 1)` or `apply(A, B)` which automatically adjusts the prime levels assuming the input MPOs have pairs of primed and unprimed indices.",
    )
  end
  N = length(A)
  N != length(B) &&
    throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
  # Special case for a single site
  N == 1 && return typeof(B)([A[1] * B[1]])
  A = orthogonalize(A, 1)
  B = orthogonalize(B, 1)
  A = sim(linkinds, A)
  sA = siteinds(uniqueinds, A, B)
  sB = siteinds(uniqueinds, B, A)
  C = typeof(B)(N)
  lCᵢ = Index[]
  R = ITensor(true)
  for i in 1:(N - 2)
    RABᵢ = R * A[i] * B[i]
    left_inds = [sA[i]..., sB[i]..., lCᵢ...]
    C[i], R = factorize(
      RABᵢ,
      left_inds;
      ortho="left",
      tags=commontags(linkinds(A, i)),
      cutoff,
      maxdim,
      mindim,
      kwargs...,
    )
    lCᵢ = dag(commoninds(C[i], R))
  end
  i = N - 1
  RABᵢ = R * A[i] * B[i] * A[i + 1] * B[i + 1]
  left_inds = [sA[i]..., sB[i]..., lCᵢ...]
  C[N - 1], C[N] = factorize(
    RABᵢ,
    left_inds;
    ortho="right",
    tags=commontags(linkinds(A, i)),
    cutoff,
    maxdim,
    mindim,
    kwargs...,
  )
  #truncate!(C; kwargs...) #TODO truncate?
  return C, zeros(N-1,maxlinkdim(C))  # TODO SVs ?
end

