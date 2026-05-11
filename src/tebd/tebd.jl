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
    tebd(psi0::MPS, Ut::MPO, Nt::Int; normalize=true, (observer!)=nothing, dt=nothing, kwargs...)

Evolve `psi0` for `Nt` steps by applying `Ut` at each step.
Truncation parameters (`cutoff`, `maxdim`, …) are forwarded to `apply`.

After each step, if an `observer!` (an Observers.jl `DataFrame`) is provided,
`update!` is called with keyword arguments:
- `state`  — the current MPS
- `step`   — the step index `nt ∈ 1:Nt`
- `time`   — `dt * nt` (only when `dt` is provided, i.e. when called via a `tMPOParams` overload)

Observer functions need only declare the kwargs they use; unsupported ones are silently dropped.

# Example
```julia
obs = observer(
    "Z"   => (; state) -> expect(state, "Z")[halfsite(state)],
    "chi" => (; state) -> maxlinkdim(state),
)
psi_t = tebd(psi0, Ut, 50; (observer!)=obs)
obs[!, "Z"]   # Vector of ⟨Z⟩ values at each step
```
"""
function tebd(psi0::MPS, Ut::MPO, Nt::Int;
              normalize::Bool = true,
              (observer!) = nothing,
              dt = nothing,
              cutoff = 1e-10,
              maxdim = 256,
              kwargs...)
    psi_t = psi0
    p     = Progress(Nt; dt=2, showspeed=true)

    for nt in 1:Nt
        psi_t = apply(Ut, psi_t; normalize, cutoff, maxdim, kwargs...)
        if !isnothing(observer!)
            t = isnothing(dt) ? nothing : dt * nt
            update!(observer!; state=psi_t, step=nt, time=t)
        end
        next!(p; showvalues=[(:Info, "chi=$(maxlinkdim(psi_t))")])
    end
    return psi_t
end

function tebd(psi0::MPS, tp::tMPOParams, Nt::Int; kwargs...)
    Ut = build_Ut(siteinds(psi0), tp)
    arrtype = Base.typename(NDTensors.unwrap_array_type(psi0[1])).wrapper
    Ut = adapt(arrtype, Ut)
    tebd(psi0, Ut, Nt; dt=tp.dt, kwargs...)
end

"""
    tebd(N::Int, tp::tMPOParams, Nt::Int; kwargs...)

Build a product initial state of length `N` from `tp`, then evolve for `Nt` steps.
If `tp` has been adapted to a GPU backend the initial state is moved to the same device.
"""
function tebd(LL::Int, tp::tMPOParams, Nt::Int; kwargs...)
    ss      = _siteinds_from_tp(LL, tp)
    psi0    = _psi0_from_tp(ss, tp)
    arrtype = Base.typename(NDTensors.unwrap_array_type(tp.bl)).wrapper
    psi0    = adapt(arrtype, psi0)
    tebd(psi0, tp, Nt; kwargs...)
end

# ── expectation-value helpers ────────────────────────────────────────────────

"""
    tebd_ev(init, tp::tMPOParams, Nt::Int, ops::Vector{<:String}; kwargs...)

Evolve for `Nt` steps and collect half-chain expectation values of `ops` after each step.
`init` is either a chain length `N::Int` or an initial state `psi0::MPS`.
Returns `(evs::Dict, psi_t::MPS)` where `evs["chis"]` holds the bond dimension history.
"""
function tebd_ev(init, tp::tMPOParams, Nt::Int, ops::Vector{<:String}; kwargs...)
    psi0 = if init isa MPS
        init
    else
        ss = _siteinds_from_tp(init, tp)
        _psi0_from_tp(ss, tp)
    end

    obs_pairs = Pair{String,Any}[op => (; state) -> expect(state, op)[halfsite(state)] for op in ops]
    push!(obs_pairs, "chi" => (; state) -> maxlinkdim(state))
    obs = observer(obs_pairs...)

    psi_t = tebd(psi0, tp, Nt; (observer!)=obs, kwargs...)

    evs = Dict(op => obs[!, op] for op in ops)
    evs["chis"] = Vector{Int}(obs[!, "chi"])
    return evs, psi_t
end

"""
    tebd_z(init, tp::tMPOParams, Nt::Int; kwargs...)

Evolve for `Nt` steps and collect the half-chain ⟨Z⟩ after each step.
`init` is either a chain length `N::Int` or an initial state `psi0::MPS`.
Returns a `Vector{ComplexF64}` of ⟨Z⟩ values.
"""
function tebd_z(init, tp::tMPOParams, Nt::Int; kwargs...)
    obs = observer("Z" => (; state) -> expect(state, "Z")[halfsite(state)])
    tebd(init, tp, Nt; (observer!)=obs, kwargs...)
    return ComplexF64.(obs[!, "Z"])
end

function tebd_z(psi0::MPS, Ut::MPO, Nt::Int; kwargs...)
    obs = observer("Z" => (; state) -> expect(state, "Z")[halfsite(state)])
    tebd(psi0, Ut, Nt; (observer!)=obs, kwargs...)
    return ComplexF64.(obs[!, "Z"])
end