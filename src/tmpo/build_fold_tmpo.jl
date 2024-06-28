""" Contracts the two edges of an MPO, making a new MPO with length (N-2) """
function contract_edges!(TT::MPO)
    llen = length(TT)

    new_first = TT[1] * TT[2]
    combi = combiner(siteind(TT,1), siteind(TT,2), tags=tags(siteind(TT,2)))
    new_first *= combi
    #new_idx = Index(dim(combinedind(combi), tags(siteind(TT,1))))
    new_first *= combi'

    new_last = TT[end-1] * TT[end]
    combi = combiner(siteind(TT,llen-1), siteind(TT,llen), tags=tags(siteind(TT,llen-1)))
    new_last *= combi
    new_last *= combi'

    pop!(TT.data)
    popfirst!(TT.data)

    TT.data[1] = new_first
    TT.data[end] = new_last

end



""" Builds the bulk tensor for the *folded* time MPO
Returns the *unrotated* combined indices as well: vL, vR, p, p' """
function build_WWc(eH_space)

    _, Wc, _ = eH_space.data

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



""" Build a *rotated and folded* TMPO associated with exp. value starting from eH tensors of U=exp(iHt) 
(defined as a regular spatial MPO on space indices). tMPO is defined on `time_sites`

We rotate our space vectors to the *right* by 90°, ie 

```
   |p'             |L
L--o--R   =>    p--o--p'
   |p              |R
```

and contract with  the initial state `init_state` on the *left* and the operator `fold_op` on the *right*

````
             p'
         |   |   |   |
[rho0]==(W)=(W)=(W)=(W)==[operator]
         |   |   |   |
             p
````
"""




""" Given `tp` parameters and `time_sites` with Nt sites,
 builds a folded tMPO with rotated indices 
 and additional identity tensors (meant to be replaced) with trivial site indices on the first and last site.
 Final length is Nt + 2 
"""

function folded_open_tMPO(tp::tmpo_params, time_sites::Vector{<:Index})

    eH = build_expH(tp)
    folded_open_tMPO(eH, time_sites)
end

function folded_open_tMPO(eH_space::MPO, time_sites::Vector{<:Index})
    
    @assert length(eH_space) == 3

    Nsteps = length(time_sites)

    WWc, iCwL, iCwR, iCp, iCps = build_WWc(eH_space)

    # define the links of the rotated MPO 
    rot_links = [Index( dim(iCp) , "Link,rotl=$(ii-1)") for ii in 1:(Nsteps + 1)]

    # Start building the tMPO
    tMPO = MPO(Nsteps+2)

    # Rotate indices and fill the MPO
    for ii = 1:Nsteps
        tMPO[ii+1] = WWc
        tMPO[ii+1] *= delta(iCwR, time_sites[ii])
        tMPO[ii+1] *= delta(iCwL, time_sites[ii]') 
        tMPO[ii+1] *= delta(iCps, rot_links[ii]) 
        tMPO[ii+1] *= delta(iCp, rot_links[ii+1]) 
    end

    t1 = Index(1,"trivial")
    tend = sim(t1)

    tMPO[1] = ITensor(eltype(WWc)[1,0,0,1] , t1, rot_links[1], t1')
    tMPO[end] = ITensor(eltype(WWc)[1,0,0,1] , tend, rot_links[end], tend')

    return tMPO 

end



""" Simple case, we feed an initial state and a fold_operator as (folded) vectors 
and build the folded tMPO from those"""
function folded_tMPO(eH_space::MPO, init_state::Vector{<:Number}, fold_op::Vector{<:Number}, time_sites::Vector{<:Index})
    
    rho0 = init_state 

    tMPO = folded_open_tMPO(eH_space, time_sites)

    if length(init_state) != linkdims(tMPO)[1]  # we already have a folded 
        if length(init_state) == linkdims(tMPO)[1] ÷ 2
            rho0 = (init_state) * (init_state')
        else
            @error "Dimension of init_state is $(length(init_state)) vs linkdim $(linkdims(tMPO)[1])"
        end
    end
    
    @assert length(fold_op) == linkdims(tMPO)[end]

    init_state_tensor = ITensor(rho0, inds(tMPO[1])...)
    fold_op_tensor = ITensor(fold_op, inds(tMPO[end])...)

    tMPO.data[1] = init_state_tensor
    tMPO.data[end] = fold_op_tensor

    contract_edges!(tMPO)

    return tMPO

end


function folded_tMPO(tp::tmpo_params, time_sites::Vector{<:Index};
     init_state = nothing, fold_op = nothing)
 
    if isnothing(init_state) 
         init_state = tp.bl 
    end 

    if isnothing(fold_op) 
        fold_op = tp.tr 
    end

    eH = build_expH(tp)
    folded_tMPO(eH, init_state, fold_op, time_sites)

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

function build_folded_left_tMPS(tp::tmpo_params, time_sites::Vector{<:Index})

    eH = build_expH(tp)

    #@info "using $(build_expH_function)"
    build_folded_left_tMPS(eH, tp.init_state, time_sites)
end


