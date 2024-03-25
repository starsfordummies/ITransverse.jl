function init_cone_ising(p::pparams, build_expH_function::Function=build_expH_ising_murg)
    time_sites = siteinds("S=3/2", 1)
    time_sites= addtags(time_sites, "time_fold")

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    eH = build_expH_function(space_sites, p)
    fold_id = ComplexF64[1,0,0,1]
    cone_mps = seed_cone_left(eH, p.init_state)

    return cone_mps

end

""" Seeds the *left* light cone temporal MPS, given `eH` MPO tensors and `init_state`,
builds a (length 1) tMPS. The `init_state` goes to the *right* """
function seed_cone_left(eH::MPO, init_state::Vector{ComplexF64})
    
    Wl = eH[1]

    iL, iR = linkinds(eH)
    iP1 = siteind(eH, 1)
    iP2 = siteind(eH, 2) 

    WWl = Wl * dag(prime(Wl,2))

    # Combine indices appropriately 
    CwR = combiner(iL,iL''; tags="cwR")
    # we flip the p<>* legs on the backwards, shouldn't be necessary if we have p<>p*
    Cp = combiner(iP1,iP1'''; tags="cp")
    Cps = combiner(iP1',iP1''; tags="cps")

    WWl = WWl * CwR * Cp * Cps

    # (in rotated indices, we trace to the left and contract with the initial state to the right 
    WWl_L = WWl * dag(Cps) * delta(iP1',iP1'')


    fold_init_state = init_state * init_state'
    init_tensor = ITensor(fold_init_state, combinedind(Cp))

    WWl_LR = WWl_L * init_tensor


    # WWc = Wc * dag(prime(Wc,2))

    # # Combine indices appropriately 
    # CwL = combiner(iL,iL''; tags="cwL")
    # CwR = combiner(iR,iR''; tags="cwR")
    # # we flip the p<>* legs on the backwards, shouldn't be necessary if we have p<>p*
    # Cp = combiner(iP2,iP2'''; tags="cp")
    # Cps = combiner(iP2',iP2''; tags="cps")

    # WWc_L = WWc * dag(Cps) * delta(iP2',iP2'')


    # iCwR = combinedind(CwR)
    # iCp = combinedind(Cp)
    # iCps = combinedind(Cps)


    tMPS = MPS(1)

    time_sites = siteinds("S=3/2", 1)
    time_sites= addtags(time_sites, "time_fold")

    tMPS[1] = WWl_LR * delta(combinedind(CwR), time_sites[1]) 

return tMPS

end


function evolve_cone(psi::MPS, nsteps::Int, 
    op::Vector{ComplexF64}, 
    ising_params::pparams,
    truncp::trunc_params
    )

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []
    chis = []
    overlaps = []

    p = Progress(nsteps; desc="$cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim)), method=$(truncp.method)", showspeed=true) 

    for dt = 1:nsteps
        #println("Evolving $dt")
        llwork = deepcopy(ll)

        # if we're worried about symmetry, evolve separately L and R 
        ll,_, ents = extend_tmps_cone_alt(llwork, rr, Id, op, ising_params, truncp)
        _,rr, ents = extend_tmps_cone_alt(llwork, rr, op, Id, ising_params, truncp)

        overlapLR = overlap_noconj(ll,rr)

        #println("lens: ", length(ll), "     ", length(rr))
        #@show (overlap_noconj(ll,rr))
        #@show maxlinkdim(ll), maxlinkdim(rr)

        #TODO  renormalize by overlap ?
        ll = ll * sqrt(1/overlapLR)
        rr = rr * sqrt(1/overlapLR)

        #println(dt)
        #println(ll)
        #println(overlap_noconj(ll,rr)/overlap_noconj(ll,ll), maxlinkdim(ll))
        push!(evs_x, expval_cone(ll, rr, ComplexF64[0,1,1,0], ising_params))
        push!(evs_z, expval_cone(ll, rr, ComplexF64[1,0,0,-1], ising_params))
        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)

        next!(p; showvalues = [(:Info,"χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])


    end

    return ll, rr, evs_x, evs_z, chis, overlaps 
end


"""
One step of the light cone algorithm: takes left and right tMPS ll, rr,
the time MPO and the operator O
extends the time MPO and the left-right tMPS by optimizing 
1) the overlap (ll1|Orr)  -> save new ll
2) the overlap (llO|1rr)  -> save new rr  (in case non symmetric)

Returns the updated left-right tMPS 
"""
function extend_tmps_cone_alt(ll::MPS, rr::MPS, 
    op_L::Vector{ComplexF64}, op_R::Vector{ComplexF64}, 
    ising_params::pparams,
    truncp::trunc_params)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    tmpo = build_ising_folded_tMPO(build_expH_ising_murg, ising_params, op_L, time_sites)
    #println("check: " , length(tmpo), length(ll), length(rr))

    psi_L = apply_extend(tmpo, ll)

    tmpo = swapprime(build_ising_folded_tMPO(build_expH_ising_murg, ising_params, op_R, time_sites), 0, 1, "Site")
    psi_R = apply_extend(tmpo, rr)

    # ! CHECK CAN WE EVER HAVE LINKS (L) != LINKS (R) ??? WHY DOES EIGEN FAIL??
    ll, rr, ents = truncate_normalize_sweep(psi_L,psi_R, truncp)

    return ll,rr,ents

end


""" Extends MPS ψ to the *left* (operator side) by one site by applying the MPO A on top of it, \\
Returns a new MPS which is the extension of ψ (extended to the left), with siteinds matching those of A
```
 | | | | | | |
 o-o-o-o-o-o-o
   | | | | | |
   o-o-o-o-o-o
``` 
"""
function apply_extend(A::MPO, ψ::MPS)

    A = sim(linkinds, A)
    ψ = sim(linkinds, ψ)
    
    @assert length(A) == length(ψ) + 1 

    N = length(ψ)

    ψ_out = MPS(N+1)

    # ! First site: is this the right way to close the MPO tensor ?
    ψ_out[1] = A[1] * ITensor(ComplexF64[1,0,0,0], siteind(A,1))

    for j in 1:N
        ψ_out[j+1] = A[j+1] * ψ[j] * delta(siteind(ψ,j), siteind(A,j+1))
    end
    
    # fix links
    for b in 1:N
        Al = commoninds(A[b], A[b + 1])
        ψl = []
        if b > 1 
            ψl = commoninds(ψ[b-1], ψ[b])
        end
        l = [Al..., ψl...]
        #println(b, l)
        if !isempty(l)
        C = combiner(l)
        ψ_out[b] *= C
        ψ_out[b + 1] *= dag(C)
        end
    end

    return noprime(ψ_out)

end




function expval_cone(ll::MPS, rr::MPS, op::Vector{ComplexF64}, ising_params::pparams)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = build_ising_folded_tMPO(build_expH_ising_murg, ising_params, fold_id, time_sites)
    psi_L = apply(tmpo, ll)

    time_sites = siteinds(rr)
    tmpo = swapprime(build_ising_folded_tMPO(build_expH_ising_murg, ising_params, op, time_sites), 0, 1, "Site")
    psi_R = apply(tmpo, rr)

    tmpo = swapprime(build_ising_folded_tMPO(build_expH_ising_murg, ising_params, fold_id, time_sites), 0, 1, "Site")
    psi_R_id = apply(tmpo, rr)


    ev = overlap_noconj(psi_L,psi_R)/overlap_noconj(psi_L,psi_R_id)

    return ev

end
