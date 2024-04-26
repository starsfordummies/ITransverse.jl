
""" New version of evolve cone using tMPO struct
TODO not implemented yet !? """
function evolve_cone_new(
    psi::MPS, 
    nsteps::Int, 
    op::Vector{ComplexF64}, 
    ising_params::pparams)

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []

    # We build 

    tmpo_id = timeMPO(ising_params, Id, 1)
    tmpo_op = timeMPO(ising_params, op, 1)


    for dt = 1:nsteps
        println("Evolving $dt")
        llwork = deepcopy(ll)


        extend_timeMPO!(tmpo_left)
        extend_timeMPO!(tmpo_right)

        # if we're worried about symmetry, evolve separately
        ll,_, ents = timestep_cone_new(llwork, rr, tmpo_left, ising_params)
        _,rr, ents = timestep_cone_new(llwork, rr, op, Id, ising_params)


        #println("lens: ", length(ll), "     ", length(rr))
        println(overlap_noconj(ll,rr))

        #TODO  renormalize by overlap ?

        #println(dt)
        #println(ll)
        #println(overlap_noconj(ll,rr)/overlap_noconj(ll,ll), maxlinkdim(ll))
        push!(evs_x, expval_cone(ll, rr, ComplexF64[0,1,1,0], ising_params))
        push!(evs_z, expval_cone(ll, rr, ComplexF64[1,0,0,-1], ising_params))

    end

    return ll, rr, evs_x, evs_z
end



function timestep_cone_new!(
    ll::MPS, rr::MPS, 
    tmpo_left::timeMPO, tmpo_right::timeMPO)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    psi_L = extend_tmps_alt(tmpo_left, ll)

    tmpo_right = swapprime(tmpo_right, 0, 1, "Site")
    psi_R = extend_tmps_alt(tmpo_right, rr)

    ll, rr, ents = truncate_normalize_sweep(psi_L,psi_R)

    return ll,rr,ents

end
