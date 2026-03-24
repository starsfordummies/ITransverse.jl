""" Truncate sweeps based on RTM """

""" 
Left truncation sweep using SVD of RTM
"""
function truncate_lsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    truncate_sweep(psi, phi; cutoff=truncp.cutoff, maxdim=truncp.maxdim)
end

function truncate_sweep(psi::MPS, phi::MPS;
        cutoff::Real  = 1e-13,
        maxdim::Int   = max(maxlinkdim(psi), maxlinkdim(phi)),
        direction::Symbol = :right,
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
    @show last_site 
    psi_ortho[last_site] = XUinv * psi_ortho[last_site]
    phi_ortho[last_site] = XVinv * phi_ortho[last_site]

    return psi_ortho, phi_ortho, SV_all
end




####### NEW SWEEPS 


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

    psiR = phi'

    workL = psiL[N]
    workR = psiR[N]

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
        workR *= psiR[jj]

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

    @show inds(phi[1])
    @show inds(workR)
    phi[1] = phi[1] * noprime(workR) # TODO primes

    return psi, phi, SV_all

end

truncate_rsweep_rtm(psi, phi; kwargs...) = truncate_rsweep_rtm!(copy(psi), copy(phi); kwargs...) 
