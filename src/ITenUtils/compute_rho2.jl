""" Compute the generalized renyi2 entropy by contracting left and right MPS"""
function rtm2_contracted(psi::MPS, phi::MPS)
    r2s = []
    for jj in eachindex(psi)[2:end-2]
        push!(r2s, rtm2_contracted(psi, phi, jj))
    end

    return r2s
end

""" At a given cut, we can compute tr(τₜ^2) as the contraction
(up-down arrows denote looped-over contracted inds)

      ^  ^  ^ 
      |  |  |
o--o--o--o--o 
|  |      
*--*--*--*--*
      |  |  |
o--o--o--o--o 
|  |      
*--*--*--*--*
      |  |  | 
      V  V  V
"""

function rtm2_contracted(psi::MPS, phi::MPS, cut::Int)
    @assert cut < length(psi)-1
    @assert cut > 1

    phi = deepcopy(phi)
    replace_siteinds!(phi, siteinds(psi))

    left1 = ITensor(1.)
    right1 = ITensor(1.)

    ind_cut_psi = linkind(psi, cut)
    ind_cut_phi = linkind(phi, cut)
    
    for jj = 1:cut
        left1 *= psi[jj] * phi[jj]
    end
    left2 = prime(left1)

    for jj = length(psi):-1:cut+1
        right1 *= psi[jj] * phi[jj]
    end

    right1 = prime(right1, ind_cut_psi)
    right2 = swapprime(right1, 1=>0)    
    
    #this is already trace(rho^2)
    tr_rho2 = left1 * right1 * left2 * right2
    
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