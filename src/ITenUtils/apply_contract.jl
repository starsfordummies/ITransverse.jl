ITensors.IndexSet(args...) = ITensors.IndexSet((i for i in args if i !== nothing)...)


""" Applies MPO to MPS from the other side (p' MPO leg, leaving open the p leg): 
instead of 

``` psi--(p)-(p)--[O]--(p')-- =  Opsi--(p)-- ```

it contracts  

``` --(p)--[O]--(p')-(p)--psi =  -(p)--Opsi ```
"""
function applys(O::MPO, psi::AbstractMPS; kwargs...)
    contract(O, prime(siteinds,psi); kwargs...)
end


""" Shorthand for simple apply(alg="naive"),  """
function applyn(O::MPO, psi::MPS; kwargs...)
    replaceprime(contractn(O, sim(linkinds, psi); kwargs...),  1 => 0)
end

function applyn(O::MPO, Q::MPO; kwargs...)
    replaceprime(contractn(O', Q; kwargs...),  2 => 1)
end

""" Shorthand for applyn + swap indices """
function applyns(O::MPO, psi::MPS; kwargs...)
    applyn(O, prime(siteinds,psi); kwargs...)
end

# TODO Check: For MPOs, applyns(A,B) = applyn(B,A) ? 



""" If we have dangling tensors at the right edge of an MPS """
function contract_dangling!(psi::AbstractMPS)
    changed = false
    while length(psi) > 1 && ndims(psi[end]) == 1 && hascommoninds(psi[end], psi[end-1]) 
        psi[end-1] = psi[end] * psi[end-1]
        pop!(psi.data)
        changed = true
    end
    
    if changed 
        ITensorMPS.reset_ortho_lims!(psi) 
    end

end


""" Copied from ITensorMPS's `contract` but adapted so that it can also extend. 
Extension is done at the right (top), ie. contraction goes like 
```
    | | | | | | |
A = o-o-o-o-o-o-o 
    | | | | |  
psi o o o o o      
```
"""
function contractn(A::MPO, ψ::AbstractMPS; preserve_tags_mps::Bool=false, kwargs...)

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


    # truncation logic. Priority is explicit kwargs over TruncParams

    truncate = get(kwargs, :truncate, false)
    cutoff = nothing
    maxdim = nothing 
    
    # If :truncp, :cutoff or :maxdim keywords are present, set truncate=true
    if haskey(kwargs, :truncp)
        (;cutoff, maxdim) = kwargs[:truncp]
        kwargs = (;kwargs..., cutoff=cutoff, maxdim=maxdim)
        truncate = true
    end

    if haskey(kwargs, :cutoff) 
        cutoff = kwargs[:cutoff]
        truncate = true
    end
    if haskey(kwargs, :maxdim)
        maxdim = kwargs[:maxdim]
        truncate = true
    end

    if truncate
        #@info "truncating, $(cutoff) - $(maxdim)"
        truncate!(ψ_out; cutoff, maxdim)
    end

    return ψ_out
end




applyd_l(A::MPO, psi::MPS; kwargs...) = noprime(contractd_l(A,psi; kwargs...))

function contractd_l(
        A::MPO,
        ψ::MPS;
        cutoff = 1.0e-13,
        maxdim = maxlinkdim(A) * maxlinkdim(ψ),
        mindim = 1,
        kwargs...,
    )::MPS

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
        ψ_out[j] = Ut
        #@info "setting $j => $(inds(Ut))"

        L = L * dag(Ut) * ψ[j+1] * A[j+1]
        simL_c = simL_c * U* ψ_c[j+1] * simA_c[j+1]

    end

    ψ_out[n] = L
    #@info "setting $n => $(inds(L))"


    for j = n+1:N 
        ψ_out[j] = A[j]
        #@info "setting $j => $(inds(A[j]))"
    end


    #@info "Setting ortho lims $(n-1):$(N+1)"
    ITensorMPS.setleftlim!(ψ_out, n-1)
    ITensorMPS.setrightlim!(ψ_out, N+1)

    return ψ_out
end

