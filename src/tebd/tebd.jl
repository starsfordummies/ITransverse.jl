""" Basic TEBD to compute the half-chain expectation value of an operator at a given time,
starts with product state |+> """
function tebd_ev(Nx::Int, tp::tMPOParams, Nt::Int, ops::Vector{<:String}, truncp::TruncParams)

    dt = 0.1


    ss =  if  hastags(tp.mp.phys_site, "S=1/2")
        siteinds("S=1/2", Nx)
    elseif  hastags(tp.mp.phys_site, "S=1")
        siteinds("S=1", Nx)
    else
        error("No good site type ? ")
    end


    eH = tp.expH_func(ss, tp.mp, tp.dt)
    #eH = build_expH_ising_murg(ss, 1.0, 0.7, 0.8, dt)

    #initial state
    psi0 = pMPS(ss, tp.bl.tensor.storage)
    psi_t = deepcopy(psi0)

    eH = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, psi0), eH)

    evs = dictfromlist(ops)

    chis = []

    LL = length(ss)

    for nt = 1:Nt
        # println("timestep N°=$(nt)\ttime=$(t)")
        psi_t = apply(eH, psi_t; maxdim = truncp.maxbondim, cutoff = truncp.cutoff, normalize = true)
        for op in keys(evs)
            push!(evs[op], expect(psi_t, op)[LL÷2])
        end
        push!(chis, maxlinkdim(psi_t))

        @info "T=$(dt*nt), chi=$(maxlinkdim(psi_t))"
    end

    evs["chis"] = chis
    return evs, psi_t

end


""" Basic TEBD to just evolve psi0 for Nt steps """
function tebd(psi0::MPS, tp::tMPOParams, Nt::Int, truncp::TruncParams)

    dt = 0.1

    eH = tp.expH_func(siteinds(psi0), tp.mp, tp.dt)
    eH = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, psi0), eH)

    psi_t = deepcopy(psi0)

    LL = length(psi0)

    @info "Evolving L=$(LL) for T=$(Nt)x$(dt)"
    @info "Params $(tp.mp)"

    for nt = 1:Nt
        # println("timestep N°=$(nt)\ttime=$(t)")
        psi_t = apply(eH, psi_t; maxdim = truncp.maxbondim, cutoff = truncp.cutoff, normalize = true)
        @info "T=$(dt*nt), chi=$(maxlinkdim(psi_t))"
    end

    return psi_t

end


""" Basic TEBD to compute the half-chain expectation value of <Z> at a given time """
function tebd_z(Nt::Int, tp::tMPOParams; truncp=TruncParams())

    #eH = build_expH_ising_murg(ss, 1.0, 0.7, 0.8, dt)

    ss = siteinds("S=1/2", 2*Nt+4)
    eH = tp.expH_func(ss, tp.mp, tp.dt)

    #initial state
    psi0 = pMPS(ss, tp.bl.tensor.storage)
    psi_t = deepcopy(psi0)

    LL = length(ss)

    for nt = 1:Nt
        # println("timestep N°=$(nt)\ttime=$(t)")
        psi_t = apply(eH, psi_t; maxdim = truncp.maxbondim, cutoff = truncp.cutoff, normalize = true)
    end
    
    return expect(psi_t, "Z")[LL÷2]

end
