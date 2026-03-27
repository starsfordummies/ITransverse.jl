""" Truncate sweeps based on RTM """

""" Our original algorithm: first bring to canonical form, then perform gauge transformations and truncations. 
Could call it "naiveRTM" ?
"""
function truncate_sweep(psi::MPS, phi::MPS;
        cutoff::Real  = 1e-13,
        maxdim::Int   = max(maxlinkdim(psi), maxlinkdim(phi)),
        direction::Symbol = :right
    )

    mpslen = length(psi)

    sweep_start, sweep_step, sweep_end, sv_offset = if direction == :right
        mpslen, -1, 2, -1
    elseif direction == :left
        1, 1, mpslen-1, 0
    else
        error("direction must be :left or :right")
    end

    psi_ortho = orthogonalize(psi, sweep_start)
    phi_ortho = orthogonalize(phi, sweep_start)

    XUinv, XVinv, env = ITensors.OneITensor(), ITensors.OneITensor(), ITensors.OneITensor()

    SV_all = zeros(Float64, mpslen-1, maxdim)

    for ii in sweep_start:sweep_step:sweep_end
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii]

        env *= Ai
        env *= Bi

        @assert order(env) == 2

        bond = ii + sv_offset  # :right → ii-1 (bond to the left), :left → ii (bond to the right)
        U, S, Vdag = svd(env, ind(env, 1); cutoff, maxdim,
                         lefttags  = tags(linkind(psi, bond)),
                         righttags = tags(linkind(phi, bond)))
        norm_factor = sum(S)

        XU    = dag(U)
        XUinv = U
        XV    = dag(Vdag)
        XVinv = Vdag

        env /= sum(S)

        env *= XU
        env *= XV

        psi_ortho[ii] = Ai * XU
        phi_ortho[ii] = Bi * XV

        Svec = collect(S.tensor.storage.data) ./ norm_factor
        SV_all[bond, 1:length(Svec)] .= Svec
    end

    last_site = sweep_end + sweep_step  # :right → 1, :left → mpslen
    #@show last_site 
    psi_ortho[last_site] = XUinv * psi_ortho[last_site]
    phi_ortho[last_site] = XVinv * phi_ortho[last_site]

    return psi_ortho, phi_ortho, SV_all
end




####### NEW SWEEPS 

# TODO direction 
function truncate_rsweep_rtm!(psi::MPS, phi::MPS; cutoff::Float64, maxdim::Int)

    @assert siteinds(psi) == siteinds(phi)
    ss = siteinds(psi)
    N = length(ss)

    SV_all = zeros(Float64, N-1, maxdim)

    psiL = psi 
    psiR = prime(linkinds, phi)


    left_env = ITensors.OneITensor()

    #elt = method == "SVD" ? Float64 : ComplexF64
    #SV_all = zeros(elt, mpslen-1, maxdim)

    Lenvs = Vector{ITensor}(undef, N-1)
    # Build left environments 
    for ii = 1:N-1 
        left_env *= psiL[ii]
        left_env *= psiR[ii]
        Lenvs[ii] = left_env
    end

    psiRp = phi'

    workL = psiL[N]
    workR = psiRp[N]

    rho = workL * Lenvs[N-1]
    rho *= workR

    F = svd(rho, ss[N]; cutoff, maxdim)

    workL *= dag(F.U)
    workR *= dag(F.V)

    Svec = collect(F.S.tensor.storage.data)/sum(F.S)  

    SV_all[N-1, 1:length(Svec)] .= Svec  

    psi[N] = F.U
    phi[N] = F.V

    for jj = reverse(2:N-1)

        workL *= psiL[jj]
        workR *= psiRp[jj]

        rho = workL * Lenvs[jj-1]
        rho *= workR

        @assert ndims(rho) == 4 

        F = svd(rho, (ss[jj], F.u); cutoff, maxdim)
        S = F.S
        workL *= dag(F.U)
        workR *= dag(F.V)

        psi[jj] = F.U
        phi[jj] = F.V

        Svec = collect(S.tensor.storage.data)/sum(S)  

        SV_all[jj-1, 1:length(Svec)] .= Svec  
    end

    psi[1] = psi[1] * workL # or work  

    # @show inds(phi[1])
    # @show inds(workR)
    phi[1] = phi[1] * noprime(workR) 

    return psi, phi, SV_all

end

truncate_rsweep_rtm(psi, phi; kwargs...) = truncate_rsweep_rtm!(copy(psi), copy(phi); kwargs...) 


""" Alternative algorithm: given two MPS, builds explicitly their RTM and truncates over it 
in a similar way to the "densitymatrix" algorithm in ITensors
"""
function truncate_sweep_rtm!(psiL::MPS, psiR::MPS;
        cutoff::Float64,
        maxdim::Int,
        direction::Symbol = :right,
        preserve_mps_tags::Bool = true
    )

    @assert siteinds(psiL) == siteinds(psiR)
    ss = siteinds(psiR)
    N  = length(ss)

    sweep_start, sweep_step, sweep_end, sv_offset = if direction == :left
        N, -1, 2, 0 
    elseif direction == :right
        1, 1, N-1, -1
    else
        error("direction must be :left or :right")
    end

    last_site = sweep_end + sweep_step  # :right → 1, :left → N

    SV_all = zeros(Float64, N-1, maxdim)

    psiR = prime(linkinds, psiR)

    # Build environments sweeping away from last_site
    envs = Vector{ITensor}(undef, N)
    env  = ITensors.OneITensor()
    for ii = sweep_start:sweep_step:sweep_end
        env *= psiL[ii]
        env *= psiR[ii]
        envs[ii] = env
    end

    #psiR = direction == :right ? phi' : prime(linkinds, phi)'
    psiR = prime(siteinds, psiR)

    workL = ITensor(1)
    workR = ITensor(1)

    Fu = nothing  # no previous bond index on first iteration

    for jj = last_site:(-sweep_step):sweep_start+sweep_step

        # On first iteration (jj == last_site) there is no env to contract
        # On subsequent iterations we accumulate workL/workR and contract with env

        workL *= psiL[jj]
        workR *= psiR[jj]

        rho = workL * envs[jj - sweep_step]
        rho *= workR
    
        svd_linds = isnothing(Fu) ? (ss[jj],) : (ss[jj], Fu)


        tsR = if preserve_mps_tags
            linkR = linkind(psiR, jj+sv_offset)
            @show linkR 
            tsR = isnothing(linkR) ? "" : tags(linkR)
        else
            "Link,l=$(jj)"
        end

        tsL = if preserve_mps_tags
            linkL  = linkind(psiL, jj+sv_offset) 
            tsL = isnothing(linkL) ? "" : tags(linkL)
        else
            "Link,l=$(jj)"
        end

        # @show Fu, isnothing(Fu)
        # @show inds(workL)
        # @show inds(workR)
        # @show inds(envs[jj - sweep_step])
        # @show inds(rho)
        
        @assert ndims(rho) == (isnothing(Fu) ? 2 : 4) "ndims=$(ndims(rho)) at jj=$jj"

        F = svd(rho, svd_linds; cutoff, maxdim, lefttags=tsL, righttags=tsR)
        Fu = F.u

        workL *= dag(F.U)
        workR *= dag(F.V)

        @debug "Setting psi[$(jj)]"
        psiL[jj] = F.U
        psiR[jj] = F.V

        Svec = collect(F.S.tensor.storage.data) ./ sum(F.S)
        SV_all[jj + sv_offset, 1:length(Svec)] .= Svec
    end

    @debug "Setting psi[$(sweep_start)]"

    psiL[sweep_start] = psiL[sweep_start] * workL
    psiR[sweep_start] = psiR[sweep_start] * workR

    return psiL, noprime(psiR), SV_all
end

truncate_sweep_rtm(psi, phi; kwargs...) = truncate_sweep_rtm!(copy(psi), copy(phi); kwargs...) 
