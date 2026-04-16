import ITransverse.ITenUtils: tcontract

""" Contract MPO-MPS with algorithm densitymatrix, starting from the left. At the end we can chop/extend 
if we work with light cone. If `contract_dangling=true` we shrink dangling edges, if present  """
function ITransverse.ITenUtils.tcontract(::Algorithm{:cudensitymatrix},
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

        L = L * dag(Ut) * ψ[j+1] * A[j+1]
        simL_c = simL_c * U* ψ_c[j+1] * simA_c[j+1]

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
        contract_dangling!(ψ_out)
    end

    return ψ_out, S_all
end
