using ITensors.Adapt: adapt

function truncate_normalize_sweep_sym(left_mps::MPS; cutoff::Float64, chi_max::Int, method::String)
    l = deepcopy(left_mps)
    truncate_normalize_sweep_sym!(l; cutoff, chi_max, method)
    #@show linkinds(l)
    return l 
end



""" 
Symmetric case: Truncates a single MPS optimizing overlap (psi (no conj) |psi)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns psi_ortho (in generalized symmetric *left* canonical form) normalized to (psi|psi)=1
and ents_sites = log(sum(σ_i)) for each site
"""
function truncate_lsweep_sym(in_psi::MPS; cutoff::Float64, chi_max::Int, method::String)

    elt = eltype(psi[1])
    XUinv= ITensor(elt, 1.)
    left_env = ITensor(elt, 1.)

    mpslen = length(psi)

    psi = orthogonalize(in_psi,1)

    ents_sites = ComplexF64[] 

    s = siteinds(psi)

    for ii = 1:mpslen-1

        Ai = XUinv * psi[ii]

        left_env *= Ai
        left_env *= Ai'
        left_env *= delta(elt, s[ii],s[ii]')


        if method == "SVD"
            F = symm_svd(left_env, ind(left_env,1), cutoff=cutoff, maxdim=chi_max)
            U = F.U
            S = F.S

            sqS = S.^(0.5)
            isqS = sqS.^(-1)
            
            XU = dag(U) * isqS
            XUinv = sqS * U

        elseif method == "EIG"
            F = symm_oeig(left_env, ind(left_env,1); cutoff)
            U = F.V
            S = F.D

            sqS = S.^(0.5)
            isqS = sqS.^(-1)

            XU = U * isqS
            XUinv = sqS * U

        end

        psi[ii] =  Ai * XU

        left_env *= XU
        left_env *= XU'

        push!(ents_sites, scalar(-S*log.(S)))
    end

    An = XUinv * psi[mpslen]

    overlap = An * An 

    if abs(scalar(overlap)) > 1e10 || abs(scalar(overlap)) < 1e-10
        @warn ("Careful! overlap overflowing? = $(scalar(overlap))")
    end

    # normalize overlap to 1 on last matrix 
    #psi[mpslen] =  An /sqrt(scalar(overlap))


    @debug "Sweep done, normalization $(overlap_noconj(psi, psi))"

    noprime!(psi) # so bad 
    # At the end, better relabeling of indices 
    for (ii,li) in enumerate(linkinds(psi))
        newlink = Index(dim(li), "Link,l=$ii")
        psi[ii] *= delta(li, newlink)
        psi[ii+1] *= delta(li, newlink)
        #@show inds(psi[ii])
    end

    return ents_sites

end



function truncate_normalize_sweep_sym_ite!(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100, method="gen_one")
    orthogonalize!(left_mps,1)
    _, _, ents = orthogonalize_gen_ents!(left_mps, length(left_mps); cutoff=svd_cutoff, normalize=true, method)
    return ents
end




""" Bring the MPS to symmetric right generalized canonical form """
function truncate_rsweep_sym(in_psi::MPS; cutoff, chi_max, method::String)

    mpslen = length(in_psi)
    elt = eltype(in_psi[1])
    s = siteinds(in_psi)

    # first bring to LEFT standard canonical form 
    psi_ortho = orthogonalize(in_psi, mpslen)

    XUinv= ITensor(elt,1.)
    right_env = ITensor(elt,1.)

    ents_sites = ComplexF64[]

    for ii = mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        right_env *= Ai'
        right_env *= delta(elt, s[ii], s[ii]')

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

        end

        psi_ortho[ii] = Ai * XU

        right_env *= XU 
        right_env *= XU' 

        push!(ents_sites, scalar(-S*log.(S)))
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
    return truncate_normalize_sweep_sym(temp; cutoff=1e-20, chi_max=2*maxlinkdim(in_mps), method="EIG")
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_gen, _ = truncate_normalize_rsweep_sym(in_mps; cutoff=1e-20, chi_max=2*maxlinkdim(in_mps), method="EIG")
    return psi_gen
end
