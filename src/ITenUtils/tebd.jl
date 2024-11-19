""" Basic TEBD to compute the expectation value of an operator at a given time"""
function tebd_ev(ss::Vector{<:Index}, tp::tMPOParams, Nt::Int, ops::Array{<:String})

    dt = 0.1

    eH = build_expH_ising_murg(ss, 1.0, 0.7, 0.8, dt)

    #initial state
    psi0 = productMPS(ss, "+")
    psi_t = deepcopy(psi0)

    evs = dictfromlist(ops)

    for nt = 1:Nt
        # println("timestep N°=$(nt)\ttime=$(t)")
        psi_t = apply(eH, psi_t; maxdim = 512, cutoff = 1e-12, normalize = true)
        for op in keys(evs)
            push!(evs[op], expect(psi_t, op)[LL÷2])
        end

        @info "T=$(dt*nt), chi=$(maxlinkdim(psi_t))"
    end

    return evs 

end