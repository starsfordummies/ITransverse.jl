""" Contracts and build/truncates over RTM """ 
function tlrcontract(
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim = 1,
        kwargs...,
    )

    @assert length(ψL) == length(ψR)
    @assert length(AL) >= length(ψL)
    @assert length(AL) == length(AR)

    ### Indices prime contractions conventions
    # 
    # ψL--p'--AL--p--     --p'--AR--p--ψR 

    sR = firstsiteinds(AR, plev=1)
    sL = firstsiteinds(AL, plev=0)

    @assert noprime.(sR) == sL

    @assert firstsiteinds(AR, plev=0)[1:length(ψR)] == noprime.(firstsiteinds(AL, plev=1)[1:length(ψL)])

    N = length(AR)
    n = length(ψR)

    mindim = 1
    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    AL = swapprime(AL, 0 => 1, tags="Site")
    ALp = replaceprime(AL, 1 => 2)
    sLp = firstsiteinds(ALp, plev=2)

    # Store the right environment tensors
    E = Vector{ITensor}(undef, N-1)

    E[1] = ψR[1] * AR[1] * AL[1] * ψL[1]

    for j in 2:n
        E[j] = E[j - 1] * ψR[j] * AR[j] * AL[j] * ψL[j]
    end
    for j in n+1:N-1 # only need N-1
        E[j] = E[j - 1] * AR[j] * AL[j]
    end

    inds.(E)

    S_all = zeros(Float64, N-1, requested_maxdim)

    R, L = if N > n 
        AR[N], ALp[N]
    else
        ψR[N] * AR[N], ψL[N] * ALp[N]
    end

    R 
    L 
    l_renorm = nothing
    r_renorm = nothing

    for j = reverse(n+1:N-1)

        # Determine smallest maxdim to use
        ciR = commoninds(R, E[j])
        ciL = commoninds(L, E[j])

        maxdim = min(dim(ciR), dim(ciL), requested_maxdim)

        rho = E[j] * L * R

        l = linkind(ψR, j)

        ts = isnothing(l) ? "" : tags(l)

        Ris = isnothing(r_renorm) ? IndexSet(sR[j+1]) : IndexSet(sR[j+1], r_renorm)
        #Lis = isnothing(l_renorm) ? IndexSet(sLp[j+1]) : IndexSet(sLp[j+1], l_renorm)

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        @show inds(rho)

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts, kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm, l_renorm = F.u, F.v

        U
        V
        # @show inds(U)
        # @show inds(Ut)
        # @show l_renorm, r_renorm

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = R * dag(U) * AR[j]
        L = L * dag(V) * ALp[j] 

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        S_all[j, 1:length(Svec)] .= Svec  
        #@show sum(Dvec)
    
    end


    for j = reverse(1:n)

        # Determine smallest maxdim to use
        cipR = commoninds(ψR[j], E[j])
        ciAR = commoninds(AR[j], E[j])
        cipL = commoninds(ψL[j], E[j])
        ciAL = commoninds(ALp[j], E[j])
        prod_dimsR = dim(cipR) * dim(ciAR)
        prod_dimsL = dim(cipL) * dim(ciAL)

        maxdim = min(prod_dimsL, prod_dimsR, requested_maxdim)

        # sR = siteinds(uniqueinds, AR, ψR, j+1)
        # sL = siteinds(uniqueinds, AL, ψL, j+1)

        rho = E[j] * L * R

        l = linkind(ψR, j)

        ts = isnothing(l) ? "" : tags(l)

        Ris = isnothing(r_renorm) ? sR[j+1] : IndexSet(sR[j+1], r_renorm)
        Lis = isnothing(l_renorm) ? sLp[j+1] : IndexSet(sLp[j+1], l_renorm)

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        @show inds(rho)
        @show Lis

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts , kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm, l_renorm = F.u, F.v

        # @show inds(U)
        # @show inds(Ut)
        # @show l_renorm, r_renorm

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = R * dag(U) * ψR[j] * AR[j]
        L = L * dag(V) * ψL[j] * ALp[j]

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        S_all[j, 1:length(Svec)] .= Svec  
        #@show sum(Dvec)

end


    ψR_out[1] = R
    ψL_out[1] = L

    return ψR_out, ψL_out, S_all 
end


#### NEW LR apply 
function tlrapply(alg, psiL::MPS, OL::MPO, OR::MPO, psiR::MPS; kwargs...)
    tpsiL, tpsiR, sv = tlrcontract(alg, psiL, OL, OR, psiR; kwargs...)
    return replaceprime(tpsiL,  1 => 0), replaceprime(tpsiR,  1 => 0), sv
end