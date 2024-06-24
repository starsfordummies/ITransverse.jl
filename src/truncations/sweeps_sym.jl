using ITensors.Adapt: adapt

function truncate_normalize_sweep_sym(left_mps::MPS; cutoff::Float64, chi_max::Int, method::String)
    l = deepcopy(left_mps)
    truncate_normalize_sweep_sym!(l; cutoff, chi_max, method)
    #@show linkinds(l)
    return l 
end



""" 
Symmetric case: Truncates a single MPS optimizing overlap (L|L) (no conj)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns L_ortho (in generalized symmetric *left* canonical form) normalized to (L|L)=1
and ents_sites = log(sum(σ_i)) for each site
"""
function truncate_normalize_sweep_sym!(left_mps::MPS; cutoff::Float64, chi_max::Int, method::String)

    XUinv= ITensor(eltype(left_mps[1]), 1.)
    left_env = ITensor(eltype(left_mps[1]), 1.)
    #Ai = ITensor(eltype(left_mps[1]), 1.)

    mpslen = length(left_mps)

    orthogonalize!(left_mps,1)

    ents_sites = [] 

    s = siteinds(left_mps)

    for ii = 1:mpslen-1

        Ai = XUinv * left_mps[ii]

        left_env *= Ai
        left_env *= Ai'
        left_env *= delta(eltype(Ai), s[ii],s[ii]')

        # TODO normalization of left_envs here ? 

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

        left_mps[ii] =  Ai * XU

        left_env *= XU
        left_env *= XU'

        push!(ents_sites, log(sum(S)))
    end

    An = XUinv * left_mps[mpslen]

    overlap = An * An 

    if abs(scalar(overlap)) > 1e10 || abs(scalar(overlap)) < 1e-10
        @warn ("Careful! overlap overflowing? = $(scalar(overlap))")
    end

    # normalize overlap to 1 on last matrix 
    left_mps[mpslen] =  An /sqrt(scalar(overlap))


    @debug "Sweep done, normalization $(overlap_noconj(left_mps, left_mps))"

    noprime!(left_mps) # so bad 
    # At the end, better relabeling of indices 
    for (ii,li) in enumerate(linkinds(left_mps))
        newlink = Index(dim(li), "Link,l=$ii")
        left_mps[ii] *= delta(li, newlink)
        left_mps[ii+1] *= delta(li, newlink)
        #@show inds(left_mps[ii])
    end

    # for ii in eachindex(left_mps)
    #     replacetags!(left_mps[ii], "v" => "Link,v=$ii")
    # end

    #@show linkinds(left_mps)

    return ents_sites

end



function truncate_normalize_sweep_sym_ite!(left_mps::MPS; svd_cutoff::Real=1e-12, chi_max::Int=100, method="gen_one")
    orthogonalize!(left_mps,1)
    _, _, ents = orthogonalize_gen_ents!(left_mps, length(left_mps); cutoff=svd_cutoff, normalize=true, method)
    return ents
end




""" Bring the MPS to symmetric right generalized canonical form """
function truncate_normalize_rsweep_sym(psi::MPS; cutoff::Real=1e-12, chi_max::Int=100, method::String)

    mpslen = length(psi)

    # bring to LEFT standard canonical form 
    psi_ortho = orthogonalize(psi, mpslen)
    s = siteinds(psi)

    XUinv= ITensor(eltype(psi[1]),1.)
    right_env = ITensor(eltype(psi[1]),1.)

    ents_sites = ComplexF64[]

    for ii = mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]

        right_env *= Ai
        right_env *= Ai'
        right_env *= delta(eltype(Ai), s[ii], s[ii]')

        # TODO normalize right_env here ?

        @assert order(right_env) == 2

        F = symm_oeig(right_env, ind(right_env,1); cutoff)
        #@show dump(F)
        U = F.V
        S = F.D

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U

        psi_ortho[ii] = Ai * XU

        right_env *= XU 
        right_env *= XU' 

        push!(ents_sites, log(sum(S)))
    end

    # the last one 
    An = XUinv * psi_ortho[1]

    overlap = An * An 

    
    @assert order(overlap) == 0 
    # normalize overlap to 1 at each step 
    psi_ortho[1] =  An /sqrt(scalar(overlap))


    return psi_ortho, ents_sites

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
