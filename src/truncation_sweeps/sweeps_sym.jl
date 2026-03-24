"""
Symmetric truncate for MPS optimizing the RTM |psi*><psi|.
direction = :left  → sweeps 1→N
direction = :right → sweeps N→1
"""
function truncate_sweep_sym(in_psi::MPS; cutoff::Float64, maxdim::Int,
                             method::String, direction::Symbol=:right)

    mpslen = length(in_psi)
    elt = method == "SVD" ? Float64 : ComplexF64
    SV_all = zeros(elt, mpslen-1, maxdim)

    sweep_start, sweep_step, sweep_end, sv_offset = if direction == :right
        mpslen, -1, 2, -1
    elseif direction == :left
        1, 1, mpslen-1, 0
    else
        error("direction must be :left or :right")
    end

    psi = orthogonalize(in_psi, sweep_start)
    sweep = sweep_start:sweep_step:sweep_end
    last_site = sweep_end + sweep_step
    ss = siteinds(psi)        # site index before any priming

    XUinv = ITensors.OneITensor()
    env = ITensors.OneITensor()

    for ii in sweep
        Ai = XUinv * psi[ii]

        env *= Ai
        env *= noprime(Ai', ss[ii]')
        @assert order(env) == 2 "unexpected env indices: $(inds(env))"

        if method == "SVD"
            F = symm_svd(env, ind(env, 1); cutoff, maxdim, lefttags="Link,l=$(ii+sv_offset)")
            U, S = F.U, F.S

            XU    = dag(U)
            XUinv = U

            env /= sum(S)

        elseif method == "EIG"
            F = symm_oeig(env, ind(env, 1); cutoff, maxdim)
            U, S = F.V, F.D

            sqS  = S .^ 0.5
            isqS = sqS .^ -1

            XU    = U * isqS
            XUinv = sqS * U

        else
            @error "Valid methods are: SVD | EIG"
        end

        psi[ii] = Ai * XU
        @assert ndims(psi[ii]) < 4 "?? $(inds(psi[ii])) - $(inds(Ai)) $(inds(XU)) "

        env *= XU
        env *= XU'

        Svec = collect(S.tensor.storage.data) ./ sum(S)
        SV_all[ii + sv_offset, 1:length(Svec)] .= Svec
    end

    psi[last_site] = XUinv * psi[last_site]

    return psi, SV_all
end


""" Truncate <psi*|psi> by explicitly building the symmetric RTM and computing its SVD decomposition"""
function truncate_sweep_sym_rtm!(psi::MPS; direction::Symbol=:right, maxdim::Int, kwargs...)

    ss = siteinds(psi)
    N = length(ss)
    psip = prime(linkinds, psi)

    sweep_start, sweep_step, sweep_end, sv_offset = if direction == :right
        N, -1, 2, 0
    elseif direction == :left
        1, 1, N-1, -1
    else
        error("direction must be :left or :right")
    end

    last_site = sweep_end + sweep_step  # :right → 1, :left → N

    SV_all = zeros(Float64, N-1, maxdim)
    envs = Vector{ITensor}(undef, N)

    # Build environments sweeping away from last_site
    env = ITensors.OneITensor()
    for ii = sweep_start:sweep_step:sweep_end
        env *= psi[ii]
        env *= psip[ii]
        envs[ii] = env  # :right → envs[N]..envs[2], :left → envs[1]..envs[N-1]
    end

    # First step: last_site contracts with the innermost environment
    work = psi[last_site]

    rho = work * envs[sweep_end]  # :right → envs[2], :left → envs[N-1]
    rho *= work'
    @assert order(rho) == 2 "check your inds? $(inds(rho))"

    F = symm_svd(rho, ss[last_site], ss[last_site]'; maxdim, kwargs...)
    work *= dag(F.U)

    Svec = collect(F.S.tensor.storage.data) ./ sum(F.S)
    SV_all[last_site + sv_offset, 1:length(Svec)] .= Svec  # :right → bond 1, :left → bond N-1
    #@show "filling $(last_site+sv_offset)"
    psi[last_site] = F.U

    for jj = sweep_end:-sweep_step:sweep_start+sweep_step
        work *= psi[jj]

        rho = work * envs[jj - sweep_step]
        rho *= work'
        @assert order(rho) == 4 "check your inds? $(inds(rho))"

        F = symm_svd(rho, (ss[jj], F.u), (ss[jj]', F.u'); maxdim, kwargs...)
        work *= dag(F.U)
        psi[jj] = F.U

        Svec = collect(F.S.tensor.storage.data) ./ sum(F.S)
        #@show "filling SV[$(jj+sv_offset)]"
        SV_all[jj + sv_offset, 1:length(Svec)] .= Svec
    end

    psi[sweep_start] = psi[sweep_start] * work

    return psi, SV_all
end

truncate_sweep_sym_rtm(psi; kwargs...) = truncate_sweep_sym_rtm!(copy(psi); kwargs...) 


## Compat for now 

truncate_lsweep_sym(in_psi::MPS; kwargs...) = truncate_sweep_sym(in_psi; direction=:left, kwargs...) 
truncate_rsweep_sym(in_psi::MPS; kwargs...) = truncate_sweep_sym(in_psi; direction=:right, kwargs...) 


function ITenUtils.tcontract(::Algorithm"RTMsym", A::MPO, ψ::MPS; preserve_tags_mps::Bool=false, kwargs...)
    psi = apply(A, ψ; alg="naive", preserve_tags_mps, truncate=false)
    psi, svals = truncate_sweep_sym(psi; kwargs...)
end


function ITenUtils.tcontract(::Algorithm"RTMsym2", A::MPO, ψ::MPS; preserve_tags_mps::Bool=false, kwargs...)
    psi = apply(A, ψ; alg="naive", preserve_tags_mps, truncate=false)
    psi, svals = truncate_sweep_sym_rtm!(psi; kwargs...)
end

""" Contract MPO-MPS with algorithm densitymatrix, starting from the left. At the end we can chop/extend 
if we work with light cone """
function ITenUtils.tcontract(::Algorithm"RTMsym3",
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

    ψ_c = ψ''
    A_c = replaceprime(prime(A, 2), 3 => 1)
    #A_c = replaceprime(simA_c, 3 => 1)

    # Store the right environment tensors
    E = Vector{ITensor}(undef, N)

    E[N] = N > n ?  A[N] * A_c[N] : ψ[N] * A[N] * A_c[N] * ψ_c[N]

    for j in reverse(n+1:N-1)
        E[j] = E[j + 1] * A[j] * A_c[j] 
    end
    for j in reverse(2:min(N-1,n))
        E[j] = E[j + 1] * ψ[j] * A[j] * A_c[j] * ψ_c[j]

    end

    L = ψ[1] * A[1]
    #simL_c =  ψ_c[1] * simA_c[1]
    l_renorm = nothing

    S_all = zeros(Float64, n-1, maxdim)

    for j in 1:min(n-1,N-1)

        # Determine smallest maxdim to use
        cip = commoninds(ψ[j], E[j + 1])
        ciA = commoninds(A[j], E[j + 1])
        prod_dims = dim(cip) * dim(ciA)
        maxdim = min(prod_dims, requested_maxdim)

        s = siteinds(uniqueinds, A, ψ, j)
        #s̃ = siteinds(uniqueinds, simA_c, ψ_c, j)
        rho = E[j + 1] * L * L''
        l = linkind(ψ, j)
        ts = isnothing(l) ? "" : tags(l)
        Lis = isnothing(l_renorm) ? IndexSet(s...) : IndexSet(s..., l_renorm)
        #Ris = isnothing(r_renorm) ? IndexSet(s̃...) : IndexSet(s̃..., r_renorm)

        @assert ndims(rho) < 5 " $j $(inds(rho))"

        F = symm_svd(rho, Lis, Lis''; cutoff, maxdim, lefttags=ts, kwargs...)
        D, U = F.S, F.U

        l_renorm = F.u

        ψ_out[j] = U

        L = L * dag(U) * ψ[j+1] * A[j+1]

        Dvec = collect(D.tensor.storage.data)/sum(D)  
 
        S_all[j, 1:length(Dvec)] .= Dvec  
    
    end

    ψ_out[n] = L

    for j = n+1:N 
        ψ_out[j] = A[j]
    end

    return ψ_out, S_all
end


""" Generalized canonical form to diagonalize symmetric RTM |psi^*><psi| 
bringing gen. orthogonality center in `ortho_center` """
function gen_canonical(in_psi::MPS, ortho_center::Int; cutoff::Float64=1e-13)

    mpslen = length(in_psi)
    #elt = eltype(in_psi[1])
    sits = siteinds(in_psi)
    sits_prime = prime(sits)
    maxdim = maxlinkdim(in_psi)

    # first bring to LEFT standard canonical form. 
    # Shouldn't matter if we are not truncating ... 
    psi_ortho = orthogonalize(in_psi, mpslen)

    #psi_ortho = copy(in_psi)

    XUinv= ITensors.OneITensor()
    right_env = ITensors.OneITensor()

    for ii = reverse(ortho_center+1:mpslen)
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        right_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        @assert order(right_env) == 2
        #@show tags(ind(right_env,1))
        F = symm_oeig(right_env, ind(right_env,1); cutoff, maxdim, tags=tags(ind(right_env,1)))
        U = F.V
        S = F.D
        #@show inds(U)

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U

        psi_ortho[ii] = Ai * XU

        right_env *= XU 
        right_env *= XU' 

    end

    # the last one 
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]


    XUinv= ITensors.OneITensor()
    left_env = ITensors.OneITensor()

    for ii = 1:ortho_center-1
        Ai = XUinv * psi_ortho[ii]

        left_env *= Ai
        left_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        @assert order(left_env) == 2
        F = symm_oeig(left_env, ind(left_env,1); cutoff, maxdim, tags=tags(ind(left_env,1)))
        U = F.V
        S = F.D

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U


        psi_ortho[ii] = Ai * XU

        left_env *= XU 
        left_env *= XU' 

    end
    psi_ortho[ortho_center] = XUinv * psi_ortho[ortho_center]


    return noprime(linkinds, psi_ortho)

end

