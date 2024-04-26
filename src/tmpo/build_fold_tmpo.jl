ITensors.space(::SiteType"S=3/2") = 4
ITensors.state(::StateName"↑", ::SiteType"S=3/2") = [1, 0, 0, 0]
ITensors.state(::StateName"+", ::SiteType"S=3/2") = [1/2, 1/2, 1/2, 1/2]


""" Builds the (bulk) tensor for the *folded* time MPO
Returns the *unrotated* indices as well, wL, wR, p, p' """
function build_WWc(eH_space)

    _, Wc, _ = eH_space.data

    # TODO better symmetry checks maybe
    #check_symmetry_itensor_mpo(Wc) # , (wL,wR), (space_p',space_p))

    space_p = siteind(eH_space,2)

    (wL, wR) = linkinds(eH_space)

    # Build W Wdagger - put double prime on Wconj for safety
    WWc = Wc * dag(prime(Wc,2))

    # Combine indices appropriately 
    CwL = combiner(wL,wL''; tags="cwL")
    CwR = combiner(wR,wR''; tags="cwR")

    # we flip the p<>* legs on the backwards, shouldn't matter if we have p<>p*
    Cp = combiner(space_p',space_p''; tags="cp")
    Cps = combiner(space_p,space_p'''; tags="cps")

    WWc = WWc * CwL * CwR * Cp * Cps


    #Return indices as well
    iCwL = combinedind(CwL)
    iCwR = combinedind(CwR)
    iCp = combinedind(Cp)
    iCps = combinedind(Cps)

    return WWc, iCwL, iCwR, iCp, iCps
end




""" Builds folded initial guess for left tMPS 
by picking the left (spatial) edges of the MPO """
function build_folded_left_tMPS(eH::MPO, init_state::Vector, time_sites)

    Nsteps = length(time_sites)
    # TODO for the Right one it would be the same, but with (wL, p, ps) and do the same basically.. 

    Wl = eH[1]

    p = siteind(eH,1)
    ps = p'
    wR = linkind(eH,1)
    
    # Build W Wdagger - put double prime on Wlonj for safety
    WWl = Wl * dag(prime(Wl,2))
    # Combine indices appropriately 
    CwR = combiner(wR,wR''; tags="cwR")
    # we flip the p<>* legs on the backwards, shouldn't be necessary since should always have p<>p*
    Cp = combiner(p,ps''; tags="cp")
    Cps = combiner(ps,p''; tags="cps")

    WWl = WWl * CwR * Cp * Cps

    rot_phys = time_sites
    rot_links = [Index(4, "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]

    #init_state = [1, 0] 
    #fold_init_state = outer(init_state, init_state) 
    fold_init_state = init_state * init_state' 


    fold_op = ComplexF64[1, 0, 0, 1]

    iCwR = ind(CwR,1)
    iCp = ind(Cp,1)
    iCps = ind(Cps,1)


    # I already prime them the other way round so it's easier to contract them
    init_tensor = ITensor(fold_init_state, iCp)
    fin_tensor = ITensor(fold_op, iCps)


    first_tensor = (fin_tensor * WWl) * delta(iCwR, rot_phys[1])* delta(iCp, rot_links[1]) 

    list_mps = [first_tensor]

    for ii in range(2,Nsteps-1)
        push!(list_mps, WWl * delta(iCwR, rot_phys[ii]) * delta(iCp, rot_links[ii-1]) * delta(iCps, rot_links[ii]) )
    end

    last_tensor = (init_tensor * WWl) * delta(iCwR, rot_phys[Nsteps]) * delta(iCps, rot_links[Nsteps-1] )

    push!(list_mps, last_tensor)

    tMPS = MPS(list_mps)

    return tMPS

end


""" Builds *rotated* and *folded* MPO for Ising, defined on `time_sites`.
Closed with `fold_op` on the left and `init_state` to the right. 
"""
function build_ising_folded_tMPO(build_expH_function::Function, JXX::Real, hz::Real, 
    dt::Number, 
    init_state::AbstractVector,
    fold_op::AbstractVector,
    time_sites::Vector{<:Index})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)

    #@info "using $(build_expH_function)"
    build_folded_tMPO(eH, init_state, fold_op, time_sites)

end

function build_ising_folded_tMPS(build_expH_function::Function, par::pparams,
    time_sites::Vector{<:Index})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    #eH = build_expH_function(space_sites, JXX, hz, dt)
    eH = build_expH_function(space_sites, par.JXX, par.hz, par.dt)

    #@info "using $(build_expH_function)"
    build_folded_left_tMPS(eH, par.init_state, time_sites)

end

function build_ising_folded_tMPO(build_expH_function::Function, p::pparams,
    fold_op::Vector{ComplexF64},
    time_sites::Vector{<:Index})

    build_ising_folded_tMPO(build_expH_function, p.JXX, p.hz, p.dt, p.init_state, fold_op, time_sites)

end





""" Build a *rotated and folded* TMPO associated with exp. value starting from eH tensors of U=exp(iHt) 
(defined as a regular spatial MPO on space indices). tMPO is defined on `time_sites`

We rotate our space vectors to the left by 90°, ie 

```
   |p'             |R
L--o--R   =>   p'--o--p 
   |p              |L
```

and contract with the operator `fold_op` on the *left*
and the initial state `init_state` on the *right*, ie.

````
           p'
       |   |   |   |
[op]X=(W)=(W)=(W)=(W)=o [in]
       |   |   |   |
           p
````
"""
function build_folded_tMPO(eH_space::MPO, init_state::Vector, fold_op::Vector, time_sites::Vector{<:Index})
    
    @assert length(eH_space) == 3

    Nsteps = length(time_sites)

    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH_space)

    #@info "checking folded WWc symmetries"
    #check_symmetry_itensor_mpo(WWc, iCwL, iCwR, iCps, iCp) 

    fold_op_tensor = ITensor((fold_op), iCps)

    fold_psi0 = (init_state) * (init_state')
    init_state_tensor = ITensor(fold_psi0, iCp)


    # define the links of the rotated MPO 
    rot_links = [Index( dim(iCp) , "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]

    # Start building the tMPO
    tMPO = MPO(Nsteps)

    # First site: Contract with operator (iCps index) and rotate the other indices
    tMPO[1] = WWc * fold_op_tensor 
    tMPO[1] *= delta(iCwL, time_sites[1]) 
    tMPO[1] *= delta(iCwR, time_sites[1]') 
    tMPO[1] *= delta(iCp, rot_links[1])  

    # Rotate indices and fill the MPO
    for ii = 2:Nsteps-1
        tMPO[ii] = WWc
        tMPO[ii] *= delta(iCwL, time_sites[ii])
        tMPO[ii] *= delta(iCwR, time_sites[ii]') 
        tMPO[ii] *= delta(iCp, rot_links[ii]) 
        tMPO[ii] *= delta(iCps, rot_links[ii-1]) 
    end

    # Close final tensor with initial state on iCp and rotate other inds
    tMPO[end] = WWc * init_state_tensor
    tMPO[end] *= delta(iCwL, time_sites[end]) 
    tMPO[end] *= delta(iCwR, time_sites[end]') 
    tMPO[end] *= delta(iCps, rot_links[end])  

return tMPO

end




function build_folded_tMPO_regul_beta(Wc::ITensor, Wc_im::ITensor, nbeta::Int, time_sites::Vector{Index{Int64}})
    #TODO NOTIMPLEMENTEDYET
    @error "not implemented yet"
    return 0 

    @assert nbeta > 1
    @assert nbeta < time_sites - 2

    
    Nsteps = length(time_sites)
    # TODO for the Right one it would be the same, but with (wL, p, ps) and do the same basically.. 

    (wL, p, ps, wR) = inds(Wc)
    (wL_i, p_i, ps_i, wR_i) = inds(Wc_im)


    # check symmetry: p<->p' , wL <-> wR 

    if  permute(Wc, (wL, ps, p, wR)).tensor == Wc.tensor
        println("Symmetric p <->p*")
    else
        println("Warning! Wc *not* symmetric p<->p*")
    end

    if permute(Wc, (wR, p, ps, wL)).tensor == Wc.tensor
        println("Symmetric wL <->wR")
    else
        println("Wc *not* symmetric wL<->wR")
    end


    
    rot_phys = time_sites
    rot_links = [Index(2, "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]

    init_state = [1 0] 
    fin_state = [1 0]


    # I already prime them the other way round so it's easier to contract them
    ph = Index(2, "ψ0")
    init_tensor = ITensor(init_state, ph)
    fin_tensor = ITensor(fin_state, ph)


    first_tensor = fin_tensor * Wc_im * delta(p_i, ph) * delta(wL_i, rot_phys[1]') * delta(wR_i, rot_phys[1])* delta(ps_i, rot_links[1]) 

    list_mpo = [first_tensor]

    for ii = 2:nbeta 
        push!(list_mpo, Wc_im *  delta(wL_i, rot_phys[ii]') * delta(wR_i, rot_phys[ii]) * delta(p_i, rot_links[ii-1]) * delta(ps_i, rot_links[ii]) )
    end
    for ii = nbeta+1:Nsteps-nbeta
        push!(list_mpo, Wc *  delta(wL, rot_phys[ii]') * delta(wR, rot_phys[ii]) * delta(p, rot_links[ii-1]) * delta(ps, rot_links[ii]) )
    end
    for ii = Nsteps-nbeta+1:Nsteps-1
        push!(list_mpo, Wc_im *  delta(wL_i, rot_phys[ii]') * delta(wR_i, rot_phys[ii]) * delta(p_i, rot_links[ii-1]) * delta(ps_i, rot_links[ii]) )
    end

    last_tensor = init_tensor * Wc_im * delta(ps_i, ph) * delta(wL_i, rot_phys[Nsteps]') * delta(wR_i, rot_phys[Nsteps]) * delta(p_i, rot_links[Nsteps-1] )

    push!(list_mpo, last_tensor)

    tMPO = MPO(list_mpo)
    println("Mpo length: $(length(tMPO))")
    return tMPO

end

