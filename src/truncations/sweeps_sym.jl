using ITensors.Adapt: adapt

function truncate_normalize_sweep_sym(left_mps::MPS; svd_cutoff::Float64, chi_max::Int, method::String)
    l = deepcopy(left_mps)
    truncate_normalize_sweep_sym!(l; svd_cutoff, chi_max, method)
    #@show linkinds(l)
    return l 
end



""" 
Symmetric case: Truncates a single MPS optimizing overlap (L|L) (no conj)
By doing first a Right sweep to standard (right-orthogonal) canonical form 
followed by a Left sweep with truncation on the generalized SVs 
returns L_ortho, ents_sites
"""
function truncate_normalize_sweep_sym!(left_mps::MPS; cutoff::Float64, chi_max::Int, method::String)

    XUinv= ITensor(1.)
    left_env = ITensor(1.)
    Ai = ITensor(1.)

    mpslen = length(left_mps)

    orthogonalize!(left_mps,1)

    ents_sites = [] 

    s = siteinds(left_mps)

    for ii = 1:mpslen-1

        Ai = XUinv * left_mps[ii]

        left_env *= Ai

        left_env *= Ai'
        left_env *= delta(s[ii],s[ii]')

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
function sweep_sym_ortho_right(psi::MPS; cutoff::Real=1e-12, chi_max::Int=100)

    mpslen = length(psi)

    # bring to LEFT standard canonical form 
    psi_ortho = orthogonalize(psi, mpslen)
    s = siteinds(psi_ortho)

    XUinv= ITensor(1.)
    renv = ITensor(1.)

    ents_sites = Vector{Float64}()

    for ii = mpslen:-1:2
        Ai = XUinv * psi_ortho[ii]

        renv *= Ai
        renv *= Ai'
        renv *= delta(s[ii], s[ii]')

        @assert order(renv) == 2

        F = symm_oeig(renv, ind(renv,1); cutoff)
        #@show dump(F)
        U = F.V
        S = F.D

        sqS = S.^(0.5)
        isqS = sqS.^(-1)

        XU = U * isqS
        XUinv = sqS * U

        psi_ortho[ii] = Ai * XU

        renv *= XU 
        renv *= XU' 

        #push!(ents_sites, log(sum(S)))
    end

    # the last one 
    An = XUinv * psi_ortho[1]

    overlap = An * An 

    
    @assert order(overlap) == 0 
    # normalize overlap to 1 at each step 
    psi_ortho[1] =  An


    return psi_ortho, ents_sites

end





"""
Just bring the MPS to generalized *left* canonical form without truncating (as far as possible)
"""
function gen_canonical_left(in_mps::MPS)
    temp = deepcopy(in_mps)
    return truncate_normalize_sweep_sym(temp; svd_cutoff=1e-14, chi_max=2*maxlinkdim(in_mps), method="EIG")
end

"""
Just bring the MPS to generalized *right* canonical form without truncating (as far as possible)
TODO should use chi_min here to make sure ! 
"""
function gen_canonical_right(in_mps::MPS)
    psi_gen, _ = sweep_sym_ortho_right(in_mps; cutoff=1e-20, chi_max=2*maxlinkdim(in_mps))
    return psi_gen
end
