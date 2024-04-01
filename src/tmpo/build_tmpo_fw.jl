#= 
""" Builds tMPO for forward-only evolution, closes top and bottom (Loschmidt echo style)
with the `init_state` vector. No beta regularization - should be superseded by other  """
function _build_fw_tMPO(eH::MPO, init_state::Vector{ComplexF64}, time_sites::Vector{<:Index})

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
=#


""" Builds temporal MPO for forward evolution only with nbeta steps of imaginary time regularization
The structure built (Loschmidt style is)
`(init_state)-W_β-W-W-W-...-W-W_β-(init_state)`
 """
function build_fw_tMPO_regul_beta(eH::MPO, eHi::MPO, 
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


########################################################################
### With regularized edges (by a few (nbeta) steps of imag time )   ####
########################################################################

# function build_ising_fw_tMPO_regul_beta( build_expH_function::Function,
#     JXX::Real, hz::Real, 
#     dt::Number, 
#     nbeta::Int, 
#     time_sites::Vector{<:Index})


#     init_state =  ComplexF64[1,0]
#     println("No initial state specified, defaulting to $init_state")

#     build_ising_fw_tMPO_regul_beta( build_expH_function,
#     JXX, hz,
#     dt, nbeta,
#     time_sites,
#     init_state)

# end

function build_ising_fw_tMPO_regul_beta( build_expH_function::Function,
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

    build_fw_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end


# function build_potts_tMPO_regul_beta( build_expH_function::Function,
#     JXX::Real, f::Real, 
#     dt::Number, 
#     nbeta::Int,
#     time_sites::Vector{<:Index})

#     init_state = ComplexF64[1,0,0]
#     println("No initial state specified, defaulting to $init_state")

#     build_potts_tMPO_regul_beta( build_expH_function,
#     JXX, f,
#     dt, nbeta,
#     time_sites,
#     init_state)
# end


function build_potts_fw_tMPO_regul_beta( build_expH_function::Function,
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

    build_fw_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end


function build_xxmodel_fw_tMPO_regul_beta( build_expH_function::Function,
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

    build_fw_tMPO_regul_beta(eH, eHi, init_state, nbeta, time_sites)

end

