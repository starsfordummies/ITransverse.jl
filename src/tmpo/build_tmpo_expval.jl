""" Builds *rotated and UNfolded* tMPO associated with expectation value of local operator `operator`,  
    with initial state `init_state`, for a given `eH` MPO of the (unrotated) exp(H)
    and `time_sites`, which should have *even* length. 
    The structure is 
    `psi0*-W*-W*-W*--OP--W-W-W-psi0 `
"""
function build_expval_tMPO(eH::MPO, init_state::AbstractArray, operator::AbstractArray, time_sites::Vector{<:Index})
     #! need to check if this works 
    #@warn "Experimental - need to check"
    @assert length(eH) == 3
    @assert length(time_sites) % 2 == 0

    Nsteps = round(Int, length(time_sites)/2)
    
    _, Wc, _ = eH.data

    space_p = siteind(eH,2) 

    (iwL, iwR) = linkinds(eH)

    rot_links = [Index(2, "Link,rotl=$ii") for ii in 1:(length(time_sites) - 1)]

    #init_state = [1 0] 
    #fin_state = init_state

    init_tensor = ITensor(init_state, space_p)
    op_tensor = ITensor(operator, space_p', space_p'')

    #@show time_sites
    tMPO = MPO(fill(Wc,length(time_sites)))

    #close edges
    tMPO[1] *= init_tensor'
    tMPO[end] *= init_tensor

    # half chain, contract with phys operator 
    tMPO[Nsteps] *= op_tensor 
    tMPO[Nsteps] *= delta(space_p'', space_p')

    # Rotate indices 
    # TODO CHECK: 90deg counter-clockwise: 
    #
    #       po'                  Ro              pn'
    #       |                    |               |
    #  Lo---o---Ro    ==>  po'---o---po  =  Ln---o---Rn
    #       |                    |               |
    #       po                   Lo              pn
    #

    # TODO check daggers if we care about symmetries...
    for ii = 1:Nsteps
        tMPO[ii] *= delta(iwL, time_sites[ii]) * delta(iwR, time_sites[ii]') 
        tMPO[ii] = dag(tMPO[ii]) 
        tMPO[2*Nsteps+1-ii] *= delta(iwL, time_sites[2*Nsteps+1-ii]) * delta(iwR, time_sites[2*Nsteps+1-ii]')
    end

    tMPO[1] *= delta(space_p, rot_links[1] )
    for ii = 2:2*Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links[ii]) * delta(space_p', rot_links[ii-1]) 
    end
    tMPO[end] *= delta(space_p', rot_links[2*Nsteps-1] )

    return tMPO

end



function build_expval_tMPS(eH::MPO, init_state::AbstractArray, operator::AbstractArray, time_sites::Vector{<:Index})
    #! need to check if this works 
   #@warn "Experimental - need to check"
   @assert length(eH) == 3
   @assert length(time_sites) % 2 == 0

   Nsteps = round(Int, length(time_sites)/2)
   
   Wl, _, _ = eH.data

   space_p = siteind(eH,2) 

   (iwL, _) = linkinds(eH)

   rot_links = [Index(2, "Link,rot_link=$ii") for ii in 1:(length(time_sites) - 1)]

   init_tensor = ITensor(init_state, space_p)
   op_tensor = ITensor(operator, space_p', space_p'')

   #@show time_sites
   tMPS = MPS(fill(Wl,length(time_sites)))

   #close edges
   tMPS[1] *= init_tensor'
   tMPS[end] *= init_tensor

   # half chain, contract with phys operator 
   tMPS[Nsteps] *= op_tensor 
   tMPS[Nsteps] *= delta(space_p'', space_p')

   # Rotate indices 
   # TODO check daggers if we care about symmetries...
   for ii = 1:Nsteps
       tMPS[ii] *= delta(iwL, time_sites[ii])  
       tMPS[ii] = dag(tMPS[ii])
       tMPS[2*Nsteps+1-ii] *= delta(iwL, time_sites[2*Nsteps+1-ii]) 
   end

   tMPS[1] *= delta(space_p, rot_links[1] )
   for ii = 2:2*Nsteps-1
       tMPS[ii] *= delta(space_p', rot_links[ii-1]) * delta(space_p, rot_links[ii]) 
   end
   tMPS[end] *= delta(space_p', rot_links[2*Nsteps-1] )

   return tMPS

end



function build_ising_expval_tMPO( build_expH_function::Function,
    JXX::Real, hz::Real, 
    dt::Number, 
    nsteps::Int,
    init_state::AbstractArray,
    operator::AbstractArray)


    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)

    time_sites =  addtags(siteinds("S=1/2", nsteps; conserve_qns = false), "time>")
    append!(time_sites,  reverse(addtags(siteinds("S=1/2", nsteps; conserve_qns = false), "time<")))

    tMPO = build_expval_tMPO(eH, init_state, operator, time_sites)
    tMPS = build_expval_tMPS(eH, init_state, operator, time_sites)

    return tMPO, tMPS
end


function build_ising_expval_tMPO( build_expH_function::Function,
    JXX::Real, hz::Real, 
    dt::Number, 
    time_sites::Vector{<:Index},
    init_state::AbstractArray,
    operator::AbstractArray)

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)

    tMPO = build_expval_tMPO(eH, init_state, operator, time_sites)
    tMPS = build_expval_tMPS(eH, init_state, operator, time_sites)

    return tMPO, tMPS
end
