function generalized_vn_entropy_symmetric_spectrum(psiL::MPS; bring_left_gen::Bool=true, normalize_eigs::Bool=true, t)
    if bring_left_gen
        psiL = gen_canonical_left(psiL)
    end
    mpslen = length(psiL)
    #links = linkinds(psiL)
    overlap = overlap_noconj(psiL,psiL)
    if abs(1-overlap) > 1e-4 && !normalize_eigs
        @warn" overlap not 1: $(overlap)"
    end
    gen_ents = ComplexF64[]
    right_env = ITensor(1.)
    psiR = prime(linkinds, psiL)
    # Start from the operator/final state side (for me that's on the left)
    ii = t
    Ai = psiL[ii]
    Bi = psiR[ii]
        #right_env = ( Ai * Bi * right_env )
    right_env = Ai * right_env
    right_env = Bi * right_env
    @assert order(right_env) == 2
        #println(left_env)
    eigss, _ = eigen(right_env, inds(right_env)[1],inds(right_env)[2])
        #gen_ent_cut = sum(eigss.*log.(eigss))
    if normalize_eigs
            eigss = eigss/sum(eigss)
    else # If we don't normalize, warn if normalization is off
    if abs(sum(eigss) - 1.) > 0.01
                @warn "RTM not well normalized? Σeigs = 1-$(abs(sum(eigss) - 1.)) "
        end
    end
    return eigss
end
