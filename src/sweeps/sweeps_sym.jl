using ITensors.Adapt: adapt


""" 
Symmetric case: Truncates a single MPS optimizing overlap (psi (no conj) |psi)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns psi_ortho (in generalized symmetric *left* canonical form) normalized to (psi|psi)=1
and ents_sites = log(sum(S_i)) for each site
"""
function truncate_lsweep_sym(in_psi::MPS; cutoff::Float64, chi_max::Int, method::String)

    elt = eltype(in_psi[1])
    mpslen = length(in_psi)

    psi_ortho = orthogonalize(in_psi,1)
    sits = siteinds(psi_ortho)

    ents_sites = ComplexF64[] 

    XUinv= ITensor(1)
    left_env = ITensor(1)

    for ii = 1:mpslen-1

        Ai = XUinv * psi_ortho[ii]

        left_env *= Ai
        left_env *= Ai'
        left_env *= delta(elt, sits[ii],sits[ii]')


        if method == "SVD"
            F = symm_svd(left_env, ind(left_env,1), cutoff=cutoff, maxdim=chi_max)
            U = F.U
            S = F.S

            sqS = S.^(0.5)
            isqS = sqS.^(-1)
            
            XU = dag(U) * isqS
            XUinv = sqS * U

        # TODO this likely doesn't work on GPU ..
        # maybe add a NDTensors.cpu()
        elseif method == "EIG"
            F = symm_oeig(left_env, ind(left_env,1); cutoff)
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
        S = NDTensors.cpu(S./sum(S))
        push!(ents_sites, scalar(-S*log.(S)))
    end


    An = XUinv * psi_ortho[end]

    overlap = An * An 

    psi_ortho[end] = An

    if abs(scalar(overlap)) > 1e10 || abs(scalar(overlap)) < 1e-10
        @warn ("Careful! overlap overflowing? = $(scalar(overlap))")
    end

    @debug "Sweep done, normalization $(overlap_noconj(psi_ortho, psi_ortho))"

    return psi_ortho, ents_sites, overlap

end




""" Bring the MPS to symmetric right generalized canonical form """
function truncate_rsweep_sym(in_psi::MPS; cutoff::Float64, chi_max::Int, method::String)

    mpslen = length(in_psi)
    elt = eltype(in_psi[1])
    sits = siteinds(in_psi)

    # first bring to LEFT standard canonical form 
    psi_ortho = orthogonalize(in_psi, mpslen)

    XUinv= ITensor(1)
    right_env = ITensor(1)

    ents_sites = fill(zero(ComplexF64), mpslen-1)  # Float64[]

    for ii = mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        
        right_env *= Ai'
        right_env *= delta(elt, sits[ii], sits[ii]')
        
        # Alt: needs to have specified link!
        #right_env *= prime(Ai, "Link")

        #@info inds(right_env)
        @assert order(right_env) == 2

        if method == "SVD"
            F = symm_svd(right_env, ind(right_env,1), cutoff=cutoff, maxdim=chi_max)
            U = F.U
            S = F.S

            sqS = S.^(0.5)
            isqS = sqS.^(-1)
            
            XU = dag(U) * isqS
            XUinv = sqS * U

        elseif method == "EIG"
            F = symm_oeig(right_env, ind(right_env,1); cutoff)
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


        # convert to CPU to avoid headaches
        # If we build "SVD" generalized entropy, normalize them to one 
        S = NDTensors.cpu(S./sum(S))
        ents_sites[ii-1] =  scalar(-S*log.(S))
    end

    # the last one 
    An = XUinv * psi_ortho[1]

    overlap = scalar(An * An)

    # normalize overlap to 1 at the last tensor ?
    psi_ortho[1] =  An # /sqrt(scalar(overlap))


    return psi_ortho, ents_sites, overlap

end





"""
Just bring the MPS to generalized *left* canonical form without truncating (as far as possible)
"""
function gen_canonical_left(in_mps::MPS)
    temp = deepcopy(in_mps)
    psi_leftgencan, _, _ = truncate_lsweep_sym(temp; cutoff=1e-15, chi_max=2*maxlinkdim(in_mps), method="EIG")
    return psi_leftgencan
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_rightgencan, _, _ = truncate_rsweep_sym(in_mps; cutoff=1e-15, chi_max=2*maxlinkdim(in_mps), method="EIG")
    return psi_rightgencan
end
