""" Truncate sweeps based on RTM """

""" 
Left truncation sweep using SVD of RTM
"""
function truncate_lsweep(psi::MPS, phi::MPS, truncp::TruncParams)
    truncate_lsweep(psi, phi; cutoff=truncp.cutoff, maxdim=truncp.maxdim)
end

function truncate_lsweep(psi::MPS, phi::MPS; cutoff::Real, maxdim::Int)

    #elt = eltype(psi[1])
    mpslen = length(phi)

    psi_ortho = orthogonalize(psi, 1)
    phi_ortho = orthogonalize(phi, 1)

    XUinv, XVinv, left_env = (ITensors.OneITensor(),ITensors.OneITensor(),ITensors.OneITensor())

    SV_all = zeros(Float64, mpslen-1, maxdim)

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        left_env *= Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim)
        norm_factor = sum(S)

        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag

        left_env *= XU
        left_env *= XV

        psi_ortho[ii] = Ai * XU  
        phi_ortho[ii] = Bi * XV

        Svec = collect(S.tensor.storage.data)/norm_factor  
        SV_all[ii, 1:length(Svec)] .= Svec  
        #push!(ents_sites, scalar(tocpu((-S*log.(S)))))
      
    end

    # the last two 
    psi_ortho[end] = XUinv * psi_ortho[end]
    phi_ortho[end] = XVinv * phi_ortho[end]

    return psi_ortho, phi_ortho, SV_all 

end






""" Truncate sweep based on Singular value decomposition of RTM |psi><phi| 

Returns updated 
1) updated `psi`
2) updated `phi` 
3) Singular values along bipartitions matrix
"""
function truncate_rsweep(psi::MPS, phi::MPS, truncp::TruncParams; fast::Bool=false)
    truncate_rsweep(psi, phi; cutoff=truncp.cutoff, maxdim=truncp.maxdim, fast)
end

function truncate_rsweep(psi::MPS, phi::MPS; cutoff::Real=1e-13, maxdim::Int=max(maxlinkdim(psi),maxlinkdim(phi)))

    mpslen = length(psi)

    # first bring to left canonical form  
    psi_ortho = orthogonalize(psi, mpslen)
    phi_ortho = orthogonalize(phi, mpslen)

    XUinv, XVinv, right_env = (ITensors.OneITensor(), ITensors.OneITensor(), ITensors.OneITensor())
    
    # For the non-symmetric case we can only truncate with SVD (=real)
    SV_all = zeros(Float64, mpslen-1, maxdim)

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]
        Bi = XVinv * phi_ortho[ii] 

        right_env *= Ai 
        right_env *= Bi 

        @assert order(right_env) == 2

        U,S,Vdag = svd(right_env, ind(right_env,1); cutoff, maxdim, lefttags=tags(linkind(psi, ii-1)),righttags=tags(linkind(phi, ii-1)))
        norm_factor = sum(S)

        XU = dag(U)
        XUinv = U

        XV = dag(Vdag) 
        XVinv = Vdag

        # TODO do we need this 
        right_env /= norm_factor
     
        right_env *= XU
        right_env *= XV

        psi_ortho[ii] = Ai * XU  
        phi_ortho[ii] = Bi * XV

        Svec = collect(S.tensor.storage.data)/norm_factor  
        SV_all[ii-1, 1:length(Svec)] .= Svec  

    end

    psi_ortho[1] = XUinv * psi_ortho[1]
    phi_ortho[1] = XVinv * phi_ortho[1]

    return psi_ortho, phi_ortho, SV_all

end



""" Generic sweep, calls left or right according to `truncp.direction` """
function truncate_sweep(psi::MPS, phi::MPS, truncp::TruncParams; method::String="RTM") 
    (;cutoff,maxdim,direction) = truncp
    truncate_sweep(psi, phi; cutoff,maxdim,direction,method)
end


function truncate_sweep(psi::MPS, phi::MPS; cutoff::Float64, maxdim::Int, direction::String, method::String="RTM")

    if method == "RDM"
        ttruncate!(psi; cutoff,maxdim)
        ttruncate!(phi; cutoff, maxdim)
    else
        if direction == "left"
            truncate_lsweep(psi, phi; cutoff,maxdim)
        elseif direction == "right"
            truncate_rsweep(psi, phi; cutoff,maxdim)
        else
            @error "Sweep direction should be left|right"
        end
    end
end




""" Inplace version of truncate_rsweep. Modifies input MPS !
 Returns generalized SVD entropies  """
function truncate_rsweep!(psi::MPS, phi::MPS; cutoff::Real=1e-12, maxdim=nothing)

    maxdim = something(maxdim, max(maxlinkdim(psi),maxlinkdim(phi)))

    #elt = eltype(psi[1])
    mpslen = length(psi)

    # first bring to left canonical form  
    orthogonalize!(psi, mpslen)
    orthogonalize!(phi, mpslen)

    XUinv, XVinv, right_env = (ITensors.OneITensor(), ITensors.OneITensor(), ITensors.OneITensor())
    
    # For the non-symmetric case we can only truncate with SVD, so ents will be real 
    ents_sites = fill(0., mpslen-1)  # Float64[]

    # Start from the *right* side 
    for ii in mpslen:-1:2
        Ai = XUinv * psi[ii]
        Bi = XVinv * phi[ii] 

        right_env *= Ai 
        right_env *= Bi 

        @assert order(right_env) == 2

        U,S,Vdag = svd(right_env, ind(right_env,1); cutoff, maxdim)
        #U,S,Vdag = matrix_svd(right_env; cutoff=cutoff, maxdim=chi_max)
        norm_factor = sum(S)
        
        XU = dag(U) 
        XUinv =  U

        #XV = dag(Vdag) 
        XVinv = Vdag

        # right_env *= XU
        # right_env *= XV
        right_env = S/norm_factor
        # Set updated matrices
        psi[ii] = Ai * XU  
        phi[ii] = Bi * dag(Vdag) # XV


    end

    # the final two
    psi[1] = XUinv * psi[1]
    phi[1] = XVinv * phi[1]


    return ents_sites

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
