# ── internal helpers ────────────────────────────────────────────────────────

""" Build siteinds matching the physical site type recorded in `tp`. """
function _siteinds_from_tp(N::Int, tp::tMPOParams)
    if hastags(tp.mp.phys_site, "S=1/2")
        siteinds("S=1/2", N)
    elseif hastags(tp.mp.phys_site, "S=1")
        siteinds("S=1", N)
    else
        error("Unsupported site type: $(tp.mp.phys_site)")
    end
end

""" Build product initial state from `tp.bl` tensor. """
_psi0_from_tp(ss, tp::tMPOParams) = pMPS(ss, storage(tp.bl))

# ── core time evolution ──────────────────────────────────────────────────────

"""
    tebd(psi0::MPS, Ut::MPO, Nt::Int; normalize=true, callback=nothing, kwargs...)

Evolve `psi0` for `Nt` steps by applying `Ut` at each step.
Truncation parameters (`cutoff`, `maxdim`, …) are forwarded to `apply`.
An optional `callback(nt, psi_t)` is called after each step.
"""
function tebd(psi0::MPS, Ut::MPO, Nt::Int;
              normalize::Bool = true,
              callback = nothing,
              kwargs...)
    psi_t = psi0
    for nt in 1:Nt
        psi_t = apply(Ut, psi_t; normalize, kwargs...)
        isnothing(callback) || callback(nt, psi_t)
    end
    return psi_t
end

function tebd(psi0::MPS, tp::tMPOParams, Nt::Int; kwargs...)
    Ut = build_Ut(siteinds(psi0), tp)
    Ut = adapt(mapreduce(NDTensors.unwrap_array_type, promote_type, psi0), Ut)
    tebd(psi0, Ut, Nt; kwargs...)
end

"""
    tebd(N::Int, tp::tMPOParams, Nt::Int; kwargs...)

Build a product initial state of length `N` from `tp`, then evolve for `Nt` steps.
"""
function tebd(LL::Int, tp::tMPOParams, Nt::Int; kwargs...)
    ss   = _siteinds_from_tp(LL, tp)
    psi0 = _psi0_from_tp(ss, tp)
    tebd(psi0, tp, Nt; kwargs...)
end

# ── expectation-value helpers ────────────────────────────────────────────────

"""
    tebd_ev(init, tp::tMPOParams, Nt::Int, ops::Vector{<:String}; kwargs...)

Evolve for `Nt` steps and collect half-chain expectation values of `ops` after each step.
`init` is either a chain length `N::Int` or an initial state `psi0::MPS`.
Truncation and other parameters are passed as keyword arguments.
Returns `(evs::Dict, psi_t::MPS)` where `evs["chis"]` holds the bond dimension history.
"""
function tebd_ev(init, tp::tMPOParams, Nt::Int, ops::Vector{<:String}; kwargs...)
    psi0 = if init isa MPS
        init
    else
        ss = _siteinds_from_tp(init, tp)
        _psi0_from_tp(ss, tp)
    end

    evs  = dictfromlist(ops)
    chis = Int[]
    dt   = tp.dt

    function cb(nt, psi)
        for op in keys(evs)
            push!(evs[op], expect(psi, op)[halfsite(psi)])
        end
        push!(chis, maxlinkdim(psi))
        @info "T=$(dt*nt), chi=$(maxlinkdim(psi))"
    end

    psi_t = tebd(psi0, tp, Nt; callback=cb, kwargs...)
    evs["chis"] = chis
    return evs, psi_t
end

"""
    tebd_z(init, tp::tMPOParams, Nt::Int; kwargs...)

Evolve for `Nt` steps and collect the half-chain ⟨Z⟩ after each step.
`init` is either a chain length `N::Int` or an initial state `psi0::MPS`.
Returns a vector of ⟨Z⟩ values.
"""
function tebd_z(init, tp::tMPOParams, Nt::Int; kwargs...)
    evs_z = ComplexF64[]
    p     = Progress(Nt; dt=2, showspeed=true)

    function cb(nt, psi)
        zeta = expect(psi, "Z")[halfsite(psi)]
        push!(evs_z, zeta)
        next!(p; showvalues=[(:Info, "chi=$(maxlinkdim(psi)), <Z>=$(zeta)")])
    end

    tebd(init, tp, Nt; callback=cb, kwargs...)
    return evs_z
end

function tebd_z(psi0::MPS, Ut::MPO, Nt::Int; kwargs...)
    evs_z = ComplexF64[]
    p     = Progress(Nt; dt=2, showspeed=true)

    function cb(nt, psi)
        zeta = expect(psi, "Z")[halfsite(psi)]
        push!(evs_z, zeta)
        next!(p; showvalues=[(:Info, "chi=$(maxlinkdim(psi)), <Z>=$(zeta)")])
    end

    tebd(psi0, Ut, Nt; callback=cb, kwargs...)
    return evs_z
end