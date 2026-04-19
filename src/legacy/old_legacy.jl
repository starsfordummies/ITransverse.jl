function rtm2_contracted_old(psi::MPS, phi::MPS; normalize_factor::Number=1.0)
    r2s = []
    @showprogress dt=30 for jj in eachindex(psi)[1:end-1]
        push!(r2s, rtm2_contracted(psi, phi, jj; normalize_factor))
    end

    return r2s
end


function generalized_vn_entropy_symmetric_alt(psiL::MPS)
    eigs_rtm = diagonalize_rtm_symmetric_alt(psiL)
    return vn_from_matrix(eigs_rtm)
end


function fwback_tMPO_old(b::FwtMPOBlocks, time_sites::Vector{<:Index}, nbetai::Int, nfw::Int, nback::Int, nbetaf::Int; 
    mid_op = [1,0,0,1], t_op::Int=nbetai+nfw, bl::ITensor = b.tp.bl, tr = b.tp.bl)

    @info "Building fwback with $(nbetai)-$(nfw)-$(nback)-$(nbetaf) - operator at $(nbetai+nfw)"

    Ntot = length(time_sites) 
    @assert nbetai + nfw + nback + nbetaf == Ntot

    (; tp, Wc, Wc_im, rot_inds) = b
    (icL, icR, icP, icPs) = (rot_inds[:L], rot_inds[:R], rot_inds[:P], rot_inds[:Ps]) 

    elt = NDTensors.unwrap_array_type(tp.bl)

    tr = adapt(elt, to_itensor(tr, icR))
    ind_op = sim(icR, tags="op")
    ten_mid_op = adapt(elt, ITensor(mid_op, ind_op, ind_op'))

    # Make same indices for real and imag, it's easier aftwards 
    replaceinds!(Wc_im, inds(Wc_im), inds(Wc))

    time_links = [Index(dim(icL), "Link,rotl=$ii") for ii in 1:(Ntot-1)]

    tMPO =  MPO(Ntot)

    for ii = 1:nbetai
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(Wc_im, (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end
    for ii = nbetai+1:nbetai+nfw
        tMPO[ii] = replaceinds(Wc, (icP, icPs), (time_sites[ii],time_sites[ii]') )
    end

    for ii = nbetai+nfw+1:nbetai+nfw+nback
        tMPO[ii] = replaceinds(dag(Wc), (icP, icPs), (time_sites[ii]',time_sites[ii]) )  # TODO Check 
    end
    for ii = nbetai+nfw+nback+1:Ntot
        #@info "$(ii) imag"
        tMPO[ii] = replaceinds(dag(Wc_im), (icP, icPs), (time_sites[ii],time_sites[ii]'))
    end


    # Label linkinds
    # TODO phys ind of bl and tr must be first one here (in case thye're not product states)

    tMPO[1] = replaceinds(tMPO[1], (icL, icR), (ind(bl,1), time_links[1]))   

    for ii = 2:Ntot-1
        tMPO[ii] = replaceinds(tMPO[ii], (icL, icR), (time_links[ii-1],time_links[ii]))
    end

    tMPO[end] = replaceind(tMPO[end], icL => time_links[end])


    # Plug operator in the column
    cl = commonind(tMPO[t_op], tMPO[t_op+1])
    tMPO[t_op] = replaceind(contract(tMPO[t_op], ten_mid_op, cl, ind_op), ind_op' => cl)
    #@show inds(tMPO[Nt])
    
    # Contract boundary states (bottom/left and top/right)
    tMPO[1] = tMPO[1] * bl  
    tMPO[end] = tMPO[end] * tr  

    return tMPO

end



""" Contracts and build/truncates over RTM """ 
function tlrcontract_old(
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

    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    AL = swapprime(AL, 0 => 1, tags="Site")
    ALp = replaceprime(AL, 1 => 2)
    #sLp = firstsiteinds(ALp, plev=2)

    # Store the right environment tensors
    E = Vector{ITensor}(undef, N-1)

    E[1] = ψR[1] * AR[1] * AL[1] * ψL[1]

    for j in 2:n
        E[j] = E[j - 1] * ψR[j] * AR[j] * AL[j] * ψL[j]
    end
    for j in n+1:N-1 # only need N-1
        E[j] = E[j - 1] * AR[j] * AL[j]
    end

    # inds.(E)

    S_all = zeros(Float64, N-1, requested_maxdim)

    R, L = if N > n 
        AR[N], ALp[N]
    else
        ψR[N] * AR[N], ψL[N] * ALp[N]
    end

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

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        #@show inds(rho)

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts, kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm= F.u

        ψR_out[j+1] = U
        ψL_out[j+1] = V

        R = R * dag(U) * AR[j]
        L = L * dag(V) * ALp[j] 

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        S_all[j, 1:length(Svec)] .= Svec  
    
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

        rho = E[j] * L * R

        l = linkind(ψR, j)

        ts = isnothing(l) ? "" : tags(l)

        Ris = isnothing(r_renorm) ? sR[j+1] : IndexSet(sR[j+1], r_renorm)

        @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

        F = svd(rho, Ris; cutoff, maxdim, lefttags=ts , kwargs...)
        S, U, V = F.S, F.U, F.V
        r_renorm = F.u

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

    return ψL_out, ψR_out, S_all 
end



""" Original implementation of new RTM algo for :right, should work  """
function tlrcontract(::Algorithm"RTMold",
        ψL::MPS,
        AL::MPO,
        AR::MPO,
        ψR::MPS;
        cutoff = 1.0e-13,
        maxdim::Int = max(maxlinkdim(AL) * maxlinkdim(ψL), maxlinkdim(AR) * maxlinkdim(ψR)),
        mindim::Int = 1,
        preserve_mps_tags::Bool = false,
        direction = :right, # TODO no left yet 
        compute_norm = false, # no norm here
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


    # this may be incorrect if both MPOs are shrinking the MPS, but for light cone it shouldn't be an issue 
    N  = min(NL, NR)  # output MPS length 

    @assert NL >= nL
    @assert NR >= nR

    sR  = firstsiteinds(AR, plev=1)

    requested_maxdim = maxdim

    ψR_out = typeof(ψR)(N)
    ψL_out = typeof(ψL)(N)

    AL  = AL'
    ALp = replaceprime(AL, 1 => 3, tags="Site")

    ψL = ψL''

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

        @assert ndims(E[j]) < 5 
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

    # @assert ndims(L) < 4 "? $(inds(L))"

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
        # @show inds(rho)
        # @assert ndims(rho) < 5 "inds(rho) @site $j ? $(inds(rho))"

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


        # @show Ris
        # @show inds(rho)
        # @show inds(U)
        # @show inds(V)

        R = hasψR ? dag(U) * R * ψR[j] * AR[j]  : dag(U) * R * AR[j]
        L = hasψL ? dag(V) * L * ψL[j] * ALp[j] : dag(V) * L * ALp[j]

        # @show inds(R)
        # @show inds(L)

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        S_all[j, 1:length(Svec)] .= Svec
    end

    ψR_out[1] = R
    ψL_out[1] = L

    return TruncLR(ψL_out, ψR_out, S_all, 1.0)
end

