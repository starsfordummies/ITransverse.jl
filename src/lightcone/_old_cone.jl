

function _init_cone_left(eH::MPO, 
    init_state::Vector{ComplexF64}, 
    fold_op::Vector{ComplexF64}, 
    time_sites::Vector{<:Index})
    
    Nsteps = length(time_sites)

    # for now 
    @assert Nsteps == 1 

    Wl  = eH.data[1]

    # with the noprime I should be sure that I'm not picking the p' even if I messed up index
    space_p = noprime(siteinds(eH)[1][2])
    (wR, _) = linkinds(eH)


    WWc = Wl * dag(prime(Wl,2))

    # Combine indices appropriately 
    CwR = combiner(wR,wR''; tags="cwR")
    # we flip the p<>* legs on the backwards, shouldn't be necessary if we have p<>p*
    Cp = combiner(space_p,space_p'''; tags="cp")
    Cps = combiner(space_p',space_p''; tags="cps")

    WWc = WWc * CwR * Cp * Cps

    println(inds(WWc))

    fold_init_state = init_state * init_state'

    iCwR = combinedind(CwR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)


    # I already prime them the other way round so it's easier to contract them
    init_tensor = ITensor(fold_init_state, iCp)
    fin_tensor = ITensor(fold_op, iCps)

    tMPS = MPS(Nsteps)

    tMPS[1] = WWc * fin_tensor * init_tensor * delta(iCwR, time_sites[1]) 

return tMPS

end




########## OLD #############


function _extend_tmps_cone(ll::MPS, rr::MPS, 
    op_L::Vector{ComplexF64}, op_R::Vector{ComplexF64}, 
    ising_params::pparams)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    psi_L = extend_mps_factorize(ll, time_sites)
    psi_R = extend_mps_factorize(rr, time_sites)


    tmpo = build_ising_folded_tMPO(build_expH_ising_murg, ising_params, op_L, time_sites)

    psi_L = apply(tmpo, psi_L)

    # psi_R = extend_mps_factorize(psi, site_type="S=3/2", tags="time")
    # time_sites = siteinds(psi_R)

    # TODO check that I swap left-right indices right
    #swapprime(in_mpo_X, 0, 1, "Site")
    tmpo = swapprime(build_ising_folded_tMPO(build_expH_ising_murg, ising_params, op_R, time_sites), 0, 1, "Site")

    psi_R = apply(tmpo, psi_R)

    ll, rr, ents = truncate_normalize_sweep(psi_L,psi_R)


    return ll,rr,ents

end

