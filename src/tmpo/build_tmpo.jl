
""" Builds ising Forward-only temporal MPO, *WARNING* Initial state is hardcoded! """
function _build_ising_fw_tMPO( build_expH_function::Function,
    JXX::Real, hz::Real, 
    dt::Number, 
    time_sites::Vector{<:Index})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)

    init_state = ComplexF64[1, 0]
    build_fw_tMPO(eH, init_state, time_sites)
end



""" TODO better handling of initial state """
function build_fw_tMPO(eH::MPO, init_state::Vector{ComplexF64}, time_sites::Vector{<:Index})

    @assert length(eH) == 3

    Nsteps = length(time_sites)
    # The time sites will provide the (rotated) physical indices

    _, Wc, _ = eH.data
    space_p = siteind(eH,2) #noprime(siteinds(eH)[2][2])

    (wL, wR) = linkinds(eH)


    rot_links = [Index(2, "Link,rot_link=$ii") for ii in 1:(Nsteps - 1)]

    #init_state = [1 0] 
    fin_state = init_state

    init_tensor = ITensor(init_state, space_p)
    fin_tensor = ITensor(fin_state, space_p')

    tMPO = MPO(Nsteps)

    for ii = 1:Nsteps
        tMPO[ii] = Wc * delta(wL, time_sites[ii]) * delta(wR, dag(time_sites[ii])') 
    end


    tMPO[1] *= fin_tensor * delta(space_p, rot_links[1]) 

    for ii = 2:Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links[ii-1]) * delta(space_p', rot_links[ii]) 
    end

    tMPO[Nsteps] *= init_tensor * delta(space_p', rot_links[Nsteps-1]) 

    return tMPO

end

########################################################################
### With regularized edges (by a few (nbeta) steps of imag time )   ####
########################################################################

function build_ising_tMPO_regul_beta( build_expH_function::Function,
    JXX::Real, hz::Real, 
    dt::Number, 
    nbeta::Int, 
    time_sites::Vector{<:Index})


    init_state =  ComplexF64[1,0]
    println("No initial state specified, defaulting to $init_state")

    build_ising_tMPO_regul_beta( build_expH_function,
    JXX, hz,
    dt, nbeta,
    time_sites,
    init_state)

end

function build_ising_tMPO_regul_beta( build_expH_function::Function,
    JXX::Real, hz::Real, 
    dt::Number, 
    nbeta::Int, 
    time_sites::Vector{<:Index},
    init_state::Vector{ComplexF64})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)
    # Imaginary time evolution
    eHi = build_expH_function(space_sites, JXX, hz, -im*dt)

    build_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end


function build_potts_tMPO_regul_beta( build_expH_function::Function,
    JXX::Real, f::Real, 
    dt::Number, 
    nbeta::Int,
    time_sites::Vector{<:Index})

    init_state = ComplexF64[1,0,0]
    println("No initial state specified, defaulting to $init_state")

    build_potts_tMPO_regul_beta( build_expH_function,
    JXX, f,
    dt, nbeta,
    time_sites,
    init_state)
end


function build_potts_tMPO_regul_beta( build_expH_function::Function,
    JXX::Real, 
    f::Real, 
    dt::Number, 
    nbeta::Int,
    time_sites::Vector{<:Index},
    init_state::Vector{ComplexF64})

    space_sites = siteinds("S=1", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, f, dt)
    # Imaginary time evolution
    eHi = build_expH_function(space_sites, JXX, f, -im*dt)

    build_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end


function build_xxmodel_tMPO_regul_beta( build_expH_function::Function,
    JXX::Real, 
    hz::Real,
    dt::Number, 
    nbeta::Int,
    time_sites::Vector{<:Index},
    init_state::Vector{ComplexF64})

    space_sites = siteinds("S=1/2", 3; conserve_qns = false)

    # Real time evolution
    eH = build_expH_function(space_sites, JXX, hz, dt)
    # Imaginary time evolution
    eHi = build_expH_function(space_sites, JXX, hz, -im*dt)

    build_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end


""" Builds temporal MPO with nbeta steps of imaginary time regularization """
function build_tMPO_regul_beta(eH::MPO, eHi::MPO, 
    init_state::Vector{ComplexF64}, 
    nbeta::Int, 
    time_sites::Vector{<:Index})

    #@assert nbeta > 0  # TODO CHECK is it ok or do we need > 1 ?
    @assert nbeta < length(time_sites) - 2
    @assert length(eH) == length(eHi) == 3


    _, Wc, _ = eH.data
    _, Wc_im, _ = eHi.data

    # with the noprime I should be sure that I'm not picking the p' even if I messed up index
    space_p = noprime(siteinds(eH)[2][2])
    @show noprime(siteinds(eH)[2][2])
    @show siteinds(eH)

    (wL, wR) = linkinds(eH)
    (wL_i, wR_i) = linkinds(eHi)

    #println(Wc)

    Nsteps = length(time_sites)

    check_symmetry_itensor_mpo(Wc) # , (wL,wR), (space_p',space_p))

    rot_links = [Index(dim(wL), "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]

    # For Lochschmidt
    fin_state = init_state



    init_tensor = ITensor(init_state, space_p)
    fin_tensor = ITensor(fin_state, space_p')

    tMPO = MPO(Nsteps)

    for ii = 1:nbeta
        tMPO[ii] = dag(Wc_im) * delta(wL_i, time_sites[ii]) * delta(wR_i, time_sites[ii]') 
    end
    for ii = nbeta+1:Nsteps-nbeta
        tMPO[ii] = Wc * delta(wL, time_sites[ii]) * delta(wR, time_sites[ii]') 
    end
    for ii = Nsteps-nbeta+1:Nsteps
        tMPO[ii] = Wc_im * delta(wL_i, time_sites[ii]) * delta(wR_i, time_sites[ii]') 
    end


    tMPO[1] *= dag(fin_tensor) * delta(space_p, rot_links[1]) 
    for ii = 2:Nsteps-1
        tMPO[ii] *= delta(space_p, rot_links[ii-1]) * delta(space_p', rot_links[ii]) 
    end
    tMPO[Nsteps] *= init_tensor * delta(space_p', rot_links[Nsteps-1]) 

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




""" TODO better handling of initial state """
function build_expval_tMPO(eH::MPO, init_state::AbstractArray, operator::AbstractArray, time_sites::Vector{<:Index})

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



""" # TODO: needs updating 
"""
function build_left_tMPS(Wl::ITensor, time_sites)
   

    Nsteps = length(time_sites)
    # TODO for the Right one it would be the same, but with (wL, p, ps) and do the same basically.. 

    (p, ps, wR) = inds(Wl)
    
    rot_sites = time_sites
    bond_dim = 2 
    rot_links = [Index(bond_dim, "Link,rotl=$ii") for ii in 1:(Nsteps - 1)]


    init_state = [1 0] 
    fin_state = [1 0]

    # I already prime them the other way round so it's easier to contract them
    init_tensor = ITensor(init_state, ps)
    fin_tensor = ITensor(fin_state, p)

    #list_mps = Vector{ITensor}()

    first_tensor = (fin_tensor * Wl) * delta(wR, rot_sites[1]) * delta(ps, rot_links[1] )

    list_mps = [first_tensor]

    for ii in range(2,Nsteps-1)
        push!(list_mps, Wl * delta(wR, rot_sites[ii]) * delta(p, rot_links[ii-1]) * delta(ps, rot_links[ii]))
    end

    last_tensor = (init_tensor * Wl) * delta(wR, rot_sites[Nsteps]) * delta(p, rot_links[Nsteps-1] )

    push!(list_mps, last_tensor)

    left_tMPS = MPS(list_mps)

    return left_tMPS

end


