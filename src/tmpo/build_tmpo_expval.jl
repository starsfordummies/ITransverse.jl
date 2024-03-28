""" Builds (unfolded) tMPO associated with expectation value of local operator `operator`, 
with initial state `init_state`, for a given `eH` MPO of the (unrotated) exp(H)
 and `time_sites`, which should have *even* length """
function build_expval_tMPO(eH::MPO, init_state::AbstractArray, operator::AbstractArray, time_sites::Vector{<:Index})
     #! need to check if this works 
    @assert length(eH) == 3

    Nsteps = round(Int, length(time_sites)/2)
    
    _, Wc, _ = eH.data

    space_p = siteind(eH,2) #noprime(siteinds(eH)[2][2])

    (wL, wR) = linkinds(eH)

    rot_links = [Index(2, "Link,rot_link=$ii") for ii in 1:(length(time_sites) - 1)]

    #init_state = [1 0] 
    fin_state = init_state

    init_tensor = ITensor(init_state, space_p)
    op_tensor = ITensor(operator, space_p', space_p'')

    #@show time_sites
    tMPO = MPO(fill(Wc,length(time_sites)))

    #close edges
    tMPO[1] *= init_tensor
    tMPO[end] *= init_tensor'

    # half chain, contract with phys operator 
    tMPO[Nsteps] *= op_tensor 
    tMPO[Nsteps] *= delta(space_p'', space_p')

    # Rotate indices 
    # TODO check daggers if we care about symmetries...
    for ii = 1:Nsteps
        tMPO[ii] *= delta(wL, time_sites[ii]) * delta(wR, time_sites[ii]') 
        tMPO[2*Nsteps+1-ii] *= delta(wL, time_sites[2*Nsteps+1-ii]) * delta(wR, time_sites[2*Nsteps+1-ii]')
        tMPO[2*Nsteps+1-ii] = dag(tMPO[2*Nsteps+1-ii]) 
    end

    tMPO[1] *= delta(space_p', rot_links[1] )
    for ii = 2:2*Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links[ii]) * delta(space_p', rot_links[ii-1]) 
    end
    tMPO[end] *= delta(space_p, rot_links[2*Nsteps-1] )

    return tMPO

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

    build_expval_tMPO(eH, init_state, operator, time_sites)
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

    build_expval_tMPO(eH, init_state, operator, time_sites)
end






# """ # TODO: needs updating 
# """
# function build_left_tMPS(Wl::ITensor, time_sites)
   

#     Nsteps = length(time_sites)
#     # TODO for the Right one it would be the same, but with (wL, p, ps) and do the same basically.. 

#     (p, ps, wR) = inds(Wl)
    
#     rot_sites = time_sites
#     bond_dim = 2 
#     rot_links = [Index(bond_dim, "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]


#     init_state = [1 0] 
#     fin_state = [1 0]

#     # I already prime them the other way round so it's easier to contract them
#     init_tensor = ITensor(init_state, ps)
#     fin_tensor = ITensor(fin_state, p)

#     #list_mps = Vector{ITensor}()

#     first_tensor = (fin_tensor * Wl) * delta(wR, rot_sites[1]) * delta(ps, rot_links[1] )

#     list_mps = [first_tensor]

#     for ii in range(2,Nsteps-1)
#         push!(list_mps, Wl * delta(wR, rot_sites[ii]) * delta(p, rot_links[ii-1]) * delta(ps, rot_links[ii]))
#     end

#     last_tensor = (init_tensor * Wl) * delta(wR, rot_sites[Nsteps]) * delta(p, rot_links[Nsteps-1] )

#     push!(list_mps, last_tensor)

#     left_tMPS = MPS(list_mps)

#     return left_tMPS

# end


