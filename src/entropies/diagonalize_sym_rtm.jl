"""
Diagonalize RTM for a *symmetric* environment (psiL,psiL) at a given `cut`
    If we are in *left* generalized symmetric canonical form, we build *right* environments 
    for the RTM and diagonalize them:

    `   
    |  |  |  |  |   _____                ____
    D--D--D--D--D--|     |   (same λ) --|    |
                   | renv|      ≃       |    |
    D--D--D--D--D--|_____|            --|____|
    |  |  |  |  |

    `
"""
function diagonalize_rtm_left_gen_sym(psiL::MPS, cut::Int; bring_left_gen::Bool=true)

    @assert cut > 1 
    @assert cut < length(psiL)

    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4
        @warn" overlap not 1: $overlap"
    end
    
    right_env = ITensors.OneITensor()

    psiR = prime(linkinds, psiL)

    # Start from the right 
    for ii = mpslen:-1:cut
        Ai = psiL[ii]
        Bi = psiR[ii]
        right_env = Ai * right_env 
        right_env = Bi * right_env

    end

    @assert order(right_env) == 2 
    #println(left_env)
    eigss = eigvals(right_env)
    
    if abs(sum(eigss) - 1.) > 0.01
        @warn "RTM not well normalized? Σeigs-1 = $(abs(sum(eigss) - 1.)) "
    end

    return diag(matrix(eigss))
    
end






""" Given `psi` MPS, build the symmetric RTM <psibar|psi>:
Bring psi in generalized RIGHT symmetric canonical form,
then contract LEFT enviroments and diagonalize them.
Returns an array of arrays containing the eigenvalues of the RTM for each cut """
function diagonalize_rtm_right_gen_sym(psi::MPS; bring_right_gen::Bool=false, normalize_factor::Number=1.0, sort_by_largest::Bool=true)

    mpslen = length(psi)

    psi_gauged = psi/normalize_factor

    if bring_right_gen
        psi_gauged = gen_canonical_right(psi_gauged)
        psi_gauged[end] /= sqrt(overlap_noconj(psi_gauged, psi_gauged))

    end

    lenv= ITensors.OneITensor()
    s = siteinds(psi_gauged)

    eigs_rtm_t = []


    # if abs(sum(eigss) - 1.) > 0.01
    #     @warn "RTM not well normalized? Σeigs-1=$(abs(sum(eigss) - 1.)) "
    # end

    @showprogress for jj in 1:mpslen-1

        lenv *= psi_gauged[jj]
        lenv *= psi_gauged[jj]'
        lenv *= delta(s[jj],s[jj]' )

        @assert ndims(lenv) == 2 

        eigss = eigvals(lenv)

        if sort_by_largest
            eigss = sort(filter(x -> abs(x) >= 1e-20, eigss), by=abs, rev=true)
        end
        
        push!(eigs_rtm_t, eigss)

    end

    eigs_rtm_t = [ ee[abs.(ee) .> 1e-20] for ee in eigs_rtm_t ]

    
    return eigs_rtm_t
end


""" Diagonalizes the RTM by bringing psi into generalized left canonical form.
Returns a list of Nsites-1 vectors of eigenvalues """
function diagonalize_rtm_symmetric(psiL::MPS; bring_left_gen::Bool=true, normalize_eigs::Bool=true, sort_by_largest::Bool=true, cutoff::Float64=1e-12)
 
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end

    mpslen = length(psiL)
    #links = linkinds(psiL)

    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    
    eigenvalues_rtm = []
    
    right_env = ITensors.OneITensor()

    psiR = prime(linkinds, psiL)

    # Start from the right and work backwards
    for ii = mpslen:-1:2
    
        right_env = psiL[ii] * right_env 
        right_env = psiR[ii] * right_env

        @assert order(right_env) == 2 
        eigss = eigvals(right_env)
        
        if normalize_eigs
            eigss = eigss/sum(eigss) 
        else # If we don't normalize, warn if normalization is off
            if abs(sum(eigss) - 1.) > 0.01
                @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
            end
        end

        if sort_by_largest
            eigss = sort(filter(x -> abs(x) >= cutoff, eigss), by=abs, rev=true)
        end

        push!(eigenvalues_rtm, eigss)
    
    end

    return reverse(eigenvalues_rtm)
    
end
