""" 
Symmetric case: Truncates a single MPS optimizing overlap (psi (no conj) |psi)
By doing first a Right <<< sweep to standard (right-orthogonal) canonical form 
followed by a Left >>> sweep with truncation on the generalized SVs 
returns psi_ortho (in generalized symmetric *left* canonical form) normalized to (psi|psi)=1
and ents_sites = log(sum(S_i)) for each site
"""
function truncate_lsweep_sym(in_psi::MPS; cutoff::Float64, maxdim::Int, method::String)

    mpslen = length(in_psi)

    psi_ortho = orthogonalize(in_psi,1)
    sits = siteinds(psi_ortho)
    sits_prime = prime(sits)

    XUinv= ITensors.OneITensor()
    left_env = ITensors.OneITensor()

    elt = method == "SVD" ? Float64 : ComplexF64
    SV_all = zeros(elt, mpslen-1, maxdim)

    for ii = 1:mpslen-1

        Ai = XUinv * psi_ortho[ii]

        left_env *= Ai
        left_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        #left_env *= delta(elt, sits[ii],sits[ii]')


        if method == "SVD"
            F = symm_svd(left_env, ind(left_env,1); cutoff, maxdim)
            U = F.U
            S = F.S

            sqS = S.^(0.5)
            isqS = sqS.^(-1)
            
            XU = dag(U) * isqS
            XUinv = sqS * U

        # TODO this likely doesn't work on GPU ..
        # maybe add a NDTensors.cpu()
        elseif method == "EIG"
            F = symm_oeig(left_env, ind(left_env,1); cutoff, maxdim)
            U = F.V
            S = F.D

            sqS = S.^(0.5)
            isqS = sqS.^(-1)

            XU = U * isqS
            XUinv = sqS * U

        end

        psi_ortho[ii] =  Ai * XU

        left_env *= XU
        left_env *= XU'

        # If we build "SVD" generalized entropy, normalize them to one 
        # S = NDTensors.cpu(S./sum(S))
        # push!(ents_sites, scalar(-S*log.(S)))

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        SV_all[ii, 1:length(Svec)] .= Svec  
    end

    An = XUinv * psi_ortho[end]

    psi_ortho[end] = An

    @debug "Sweep done, normalization $(overlap_noconj(psi_ortho, psi_ortho))"

    return psi_ortho, SV_all

end




""" Symmetric truncate for MPS optimizing RTM |psi^*><psi| """
function truncate_rsweep_sym(in_psi::MPS; cutoff::Float64, maxdim::Int, method::String, fast::Bool=false)

    mpslen = length(in_psi)
    #elt = eltype(in_psi[1])
    sits = siteinds(in_psi)
    sits_prime = prime(sits)

    # first bring to LEFT standard canonical form 
    psi_ortho = orthogonalize(in_psi, mpslen)

    XUinv= ITensors.OneITensor()
    right_env = ITensors.OneITensor()

    elt = method == "SVD" ? Float64 : ComplexF64
    SV_all = zeros(elt, mpslen-1, maxdim)

    for ii = mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        #right_env *= Ai'
        #right_env *= delta(elt, sits[ii], sits_prime[ii])
        right_env *= replaceind(Ai', sits_prime[ii] => sits[ii])

        # Alt: needs to have specified link!
        #right_env *= prime(Ai, "Link")

        @assert order(right_env) == 2

        if method == "SVD"
            F = symm_svd(right_env, ind(right_env,1); cutoff, maxdim)
            U = F.U
            S = F.S

            XU = dag(U)
            XUinv = U

            if fast # if we don't care about putting the output psi in generalized canonical form
                right_env /= sum(S)
            else
                sqS = S.^(0.5)
                isqS = sqS.^(-1)
            
                XU = XU * isqS
                XUinv = sqS * XUinv
            end

        elseif method == "EIG"
            F = symm_oeig(right_env, ind(right_env,1); cutoff, maxdim)
            U = F.V
            S = F.D

            sqS = S.^(0.5)
            isqS = sqS.^(-1)

            XU = U * isqS
            XUinv = sqS * U

        else
            @error "Need to specify valid method: SVD|EIG"
        end

        psi_ortho[ii] = Ai * XU

        right_env *= XU 
        right_env *= XU' 

        # if we want to cheat 
        #  right_env = delta(inds(right_env))


        # If we build "SVD" generalized entropy, normalize SV to one 
        #S = S/sum(S)
        #ents_sites[ii-1] =  scalar(-S*log.(S))

        Svec = collect(S.tensor.storage.data)/sum(S)  
 
        SV_all[ii-1, 1:length(Svec)] .= Svec  
    end

    # the last one 
    An = XUinv * psi_ortho[1]

    # normalize overlap to 1 at the last tensor ?
    psi_ortho[1] =  An # /sqrt(scalar(overlap))

    return psi_ortho, SV_all

end





"""
Bring the MPS to generalized *left* canonical form without truncating (as far as possible)
This is symmetric, so we use symmetric eigenvalue decomposition, A => O D O^T with O complex orthogonal 
"""
function gen_canonical_left(in_mps::MPS)  # TODO: polar decomp?
    temp = deepcopy(in_mps)
    psi_leftgencan, _ = truncate_lsweep_sym(temp; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_leftgencan
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_rightgencan, _ = truncate_rsweep_sym(in_mps; cutoff=1e-14, maxdim=2*maxlinkdim(in_mps), method="EIG")
    return psi_rightgencan
end



function ITenUtils.tcontract(::Algorithm"RTMsym", A::MPO, ψ::MPS; preserve_tags_mps::Bool=false, kwargs...)
    psi = apply(A, ψ; alg="naive", preserve_tags_mps, truncate=false)
    psi, svals = truncate_rsweep_sym(psi; kwargs...)
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
        F = symm_oeig(right_env, ind(right_env,1); cutoff, maxdim)
        U = F.V
        S = F.D

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
        F = symm_oeig(left_env, ind(left_env,1); cutoff, maxdim)
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


    return psi_ortho

end

