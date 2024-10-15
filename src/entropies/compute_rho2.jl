""" Given two MPS `<psi` and `|phi>`, 
computes tr(τₜ^2) the trace of the reduced transition matrices |phi><psi| at the various cuts,
 by contracting left and right MPS (see also below). 
 The normalization must be given via `normalization_factor` """
function rtm2_contracted(psi::MPS, phi::MPS; normalize_factor::Number=1.0)
    r2s = []
    @showprogress for jj in eachindex(psi)[1:end-1]
        push!(r2s, rtm2_contracted(psi, phi, jj; normalize_factor))
    end

    return r2s
end

""" Same as rtm2_contracted() with normalize_factor= overlap_noconj(psi,phi)"""
function rtm2_contracted_normalized(psi::MPS, phi::MPS)
    r2s = []
    normalize_factor = overlap_noconj(psi,phi)
    normalize_factor > 1e10 || normalize_factor < 1e-10 && @warn "overlap overflow? $(normalize_factor)"
    @showprogress for jj in eachindex(psi)[1:end-1]
        push!(r2s, rtm2_contracted(psi, phi, jj; normalize_factor))
    end

    return r2s
end



""" At a given cut, we can compute tr(τₜ^2) as the contraction
(asterisks denote contracted inds, ie. we trace over topmost with lowmost inds)

```
      *  *  * 
      |  |  |
o--o--o--o--o  |phi>
|  |      
□==□==□==□==□  <psi|
      |  |  |
o--o--o--o--o  |phi>
|  |      
□==□==□==□==□  <psi|
      |  |  | 
      *  *  *
```
We do this by building the left and right blocks 

```
o--o--
|  |      =  left
□==□==
```

and 

```
            ==□==□==□  
right         |  |  |
            --o--o--o  
```

and appropriately contract them. This should basically amount to the contraction
```
tr(--[LEFT]==[RIGHT]--[LEFT]==[RIGHT]--)

```
"""
function rtm2_contracted(psi::MPS, phi::MPS, cut::Int; normalize_factor::Number=1.0)

    # valid cuts go from 1 to L-1
    @assert cut < length(psi)
    @assert cut > 0


    phi = deepcopy(phi)

    phi = sim(linkinds,phi)/normalize_factor

    #replace_siteinds!(phi, siteinds(psi))
    match_siteinds!(psi, phi)

    left = ITensor(eltype(psi[1]),1.)
    right = ITensor(eltype(psi[1]),1.)

    for jj = 1:cut
        left *= psi[jj] 
        left *= phi[jj]
    end

    for jj = length(psi):-1:cut+1
        right *= psi[jj]
        right *= phi[jj]
    end

    @assert inds(left) == inds(right)


    v_psi = linkind(psi,cut)
    right = prime(right, v_psi)
    
    #trace(rho^2) is just the product left * right * left * right as depicted above
    tr_rho2 = left * right
    #@info "2: $(inds(tr_rho2))"
    tr_rho2 *= swapprime(tr_rho2, 1=>0)
    
    return scalar(tr_rho2)
end

""" Same as before but using matrices - result should be the same, useful for debugging """
function rtm2_contracted_m(psi::MPS, phi::MPS, cut::Int; normalize_factor::Number=1.0)

    # valid cuts go from 1 to L-1
    @assert cut < length(psi)
    @assert cut > 0


    phi = deepcopy(phi)

    phi = sim(linkinds,phi)/normalize_factor

    #replace_siteinds!(phi, siteinds(psi))
    match_siteinds!(psi, phi)

    left = ITensor(eltype(psi[1]),1.)
    right = ITensor(eltype(psi[1]),1.)

    for jj = 1:cut
        left *= psi[jj] 
        left *= phi[jj]
    end

    for jj = length(psi):-1:cut+1
        right *= psi[jj]
        right *= phi[jj]
    end

    @assert inds(left) == inds(right)

    mL = Matrix(left, inds(left))
    mR = Matrix(right, reverse(inds(left)))

    return tr(mL * mR * mL * mR)
    
end


""" Given a set of eigenvalues, simply computes sum(λ_i^2) """
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
    r2 = transpose(eigenvalues) * eigenvalues
end




# Brute-force diagonalization
""" Brute-force diagonalization of the RTM built from the input MPS psi and phi. Only does it for small chains """
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

