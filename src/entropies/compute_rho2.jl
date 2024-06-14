""" Compute the generalized renyi2 entropy by contracting left and right MPS"""
function rtm2_contracted(psi::MPS, phi::MPS; normalize_factor::Number=1.0)
    r2s = []
    for jj in eachindex(psi)[1:end-1]
        push!(r2s, rtm2_contracted(psi, phi, jj; normalize_factor))
    end

    return r2s
end

""" At a given cut, we can compute tr(τₜ^2) as the contraction
(up-down arrows denote looped-over contracted inds)

      ^  ^  ^ 
      |  |  |
o--o--o--o--o  |phi>
|  |      
*--*--*--*--*  <psi|
      |  |  |
o--o--o--o--o  |phi>
|  |      
*--*--*--*--*  <psi|
      |  |  | 
      V  V  V
"""

function rtm2_contracted(psi::MPS, phi::MPS, cut::Int; normalize_factor::Number=1.0)
    @assert cut < length(psi)
    @assert cut > 0


    phi = deepcopy(phi)

    # we can do it here or at the end ..
    # if normalize
    #     phi = phi/overlap_noconj(phi,psi)
    # end
    phi = phi/normalize_factor


    replace_siteinds!(phi, siteinds(psi))

    left1 = ITensor(1.)
    right1 = ITensor(1.)

    ind_cut_psi = linkind(psi, cut)
    ind_cut_phi = linkind(phi, cut)
    
    for jj = 1:cut
        left1 *= psi[jj] 
        left1 *= phi[jj]
    end

    left2 = prime(left1)

    for jj = length(psi):-1:cut+1
        right1 *= psi[jj]
        right1 *= phi[jj]
    end

    right1 = prime(right1, ind_cut_psi)
    right2 = swapprime(right1, 1=>0)    
    
    #this is already trace(rho^2)
    tr_rho2 = left1
    #@info "1: $(inds(tr_rho2))"
    tr_rho2 *= right1 
    #@info "2: $(inds(tr_rho2))"
    tr_rho2 *= left2 
    #@info "3: $(inds(tr_rho2))"
    tr_rho2 *= right2
    
    return scalar(tr_rho2)
end



function rho2(eigenvalues::ITensor)
    meig = matrix(eigenvalues)
    if isdiag(meig)
        r2 = rho2(diag(meig))
    else
        @error "Eigenvalues matrix not diagonal ? "
        r2 = NaN
    end

    return r2
end

function rho2(eigenvalues::Vector)
    r2 = sum(eigenvalues .* eigenvalues)
end


function rho2_eigen(all_eigenvalues::Vector)
    r2s = []
    for eigenvalues in all_eigenvalues
         push!(r2s, sum(eigenvalues .* eigenvalues))
    end
end



# Brute-force diagonalization

function rtm2_bruteforce(psi::MPS, phi::MPS)

    @assert length(psi) < 14  # don't do this otherwise..

    s = siteinds(psi)
    replace_siteinds!(phi, s)
    rho = outer(dag(psi)',phi)
    ss = siteinds(rho)

    rho_contracted = ITensor(1.)
    for jj in eachindex(rho)
        rho_contracted = rho_contracted * rho[jj]
    end

    rho_t = rho_contracted
    r2s = [] 

    for jj in eachindex(rho)[1:end-1]

        rho_t = rho_t * delta(siteinds(rho,jj))
        #@show inds(rho_t)
        # merge indices 
        comb = combiner(s[jj+1:end])
        rho_t *= comb 
        rho_t *= comb'
        #@show inds(rho_t)
        #diagonalize
        vals, _ = eigen(rho_t)
        #@info "[$jj] $(rho2(diag(matrix(vals))))"
        push!(r2s, rho2(vals))
        # revert merge 
        rho_t *= comb 
        rho_t *= comb'
        #@show inds(rho_t)

    end

    return r2s
end


""" Bring psi in generalized RIGHT symmetric canonical form,
then contract LEFT enviroments and diagonalize them """
function rtm2_sym_gauged(psi::MPS, normalize_factor::Number=1.0)

    mpslen = length(psi)

    psi_gauged = gen_canonical_right(psi)

    psi_gauged = psi_gauged/normalize_factor


    lenv= ITensor(1.)
    s = siteinds(psi_gauged)

    r2s = []
    r2s_check = []
    for jj in 1:mpslen-1

        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )

        @assert ndims(lenv) == 2 

        vals, vecs = eigen(lenv, ind(lenv,1), ind(lenv,2))

        push!(r2s, rho2(vals))

        if mpslen - jj < 4
            renv = ITensor(1.)
            for kk in mpslen:-1:jj+1
                renv *= psi_gauged[kk]
                renv *= psi_gauged[kk]'
            end

            rtm_full = lenv * renv
            vals_fu, vecs_fu = eigen(rtm_full, inds(rtm_full,plev=0), inds(rtm_full,plev=1))

            push!(r2s_check, rho2(vals_fu))
        end

    end
    
    return r2s, r2s_check 
end



""" For a state psi, build the symmetric RTM <psibar|psi>,
Bring psi in generalized RIGHT symmetric canonical form,
then contract LEFT enviroments and diagonalize them.
Returns an array of arrays containing the eigenvalues of the RTM for each cut """
function diagonalize_rtm_sym_gauged(psi::MPS; normalize_factor::Number=1.0)

    mpslen = length(psi)

    psi_gauged = gen_canonical_right(psi)

    psi_gauged = psi_gauged/normalize_factor


    lenv= ITensor(1.)
    s = siteinds(psi_gauged)

    eigs_rho = []
    eigs_rho_check = []

    for jj in 1:mpslen-1

        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )

        @assert ndims(lenv) == 2 

        vals, vecs = eigen(lenv, ind(lenv,1), ind(lenv,2))

        push!(eigs_rho, vals)

        if mpslen - jj < 4
            renv = ITensor(1.)
            for kk in mpslen:-1:jj+1
                renv *= psi_gauged[kk]
                renv *= psi_gauged[kk]'
            end

            rtm_full = lenv * renv
            vals_full, _ = eigen(rtm_full, inds(rtm_full,plev=0), inds(rtm_full,plev=1))

            push!(eigs_rho_check, vals_full)
        end

    end
    
    return eigs_rho, eigs_rho_check 
end