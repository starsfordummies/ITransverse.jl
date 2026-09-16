"""
    compose_steps(compose::Symbol, dt) -> Tuple of sub-steps

Sub-step lengths of a symmetric composition of a 2nd-order `U₂`, summing to `dt`.

| `compose`   | sub-steps | factors | composite bond dim | order | error constant |
|:------------|:----------|:--------|:-------------------|:------|:---------------|
| `:none`     | `[dt]`           | 1 | `D`   | 2 | 1×   |
| `:suzuki5`  | `[p,p,1-4p,p,p]` | 5 | `D^5` | 4 | 1×   |
| `:yoshida3` | `[w,1-2w,w]`     | 3 | `D^3` | 4 | ~19× |

with `p = 1/(4-4^{1/3}) ≈ 0.4145` and `w = 1/(2-2^{1/3}) ≈ 1.3512` (Yoshida's
triple jump, Phys. Lett. A 150 (1990) 262).

Both compositions reach 4th order only if the base `U₂` is **time-self-adjoint**
(`U₂(dt)U₂(-dt) = 1`). `Murg()` is — it is an exact palindromic Strang splitting —
so for Ising/Potts the table holds. The Ghent/W-II family is not, and there both
compositions give 3rd order instead (see `DevTransverse/src/tmpo4o/fw_tmpo_4o.jl`).

## When to prefer `:yoshida3`

`:suzuki5` is the better *integrator*, and it is the default. Measured on Ising
(J=1, g=0.95, h=1.4, per-step operator norm, L=8) both reach local slope 5, i.e.
4th order, with `:yoshida3` a factor 16.8 / 18.8 / 19.3 / 19.5 worse at
`dt = 0.2 / 0.1 / 0.05 / 0.025`. Matching accuracy therefore needs `dt` smaller by
`19.5^(1/4) ≈ 2.1`, i.e. ~6.3 factor-applications per unit time against
`:suzuki5`'s 5. If all you do is *apply* the composite, `:yoshida3` loses.

`:yoshida3` wins when the composite is an **intermediate to be built and
contracted densely** rather than applied — above all as the input to
`ti_sym_compress`/`ti_sym_gate`, which contracts several sites of the exact
composite to form its truncation environment. `applyn` does not truncate, so the
composite's bond dimension really is `D^5` against `D^3` (measured on 9 sites,
`dt = 0.1`, with the wall time to build it):

| base `U₂` | `D` | `:suzuki5` | | `:yoshida3` | |
|:---|---:|---:|---:|---:|---:|
| `Murg()`    | 2 |   32 | 0.01 s |   8 | <0.01 s |
| `Ghent2Sym` | 3 |  243 | 0.05 s |  27 |  0.01 s |
| `Ghent3`    | 6 | 7776 | 26.4 s | 216 |  0.03 s |

At `D = 2` the difference is irrelevant. At `D = 6` it is 36× the bond dimension
and ~900× the build time, and decides whether the compression is feasible at all.

What it costs downstream, measured through `ti_sym_gate` on the same model:

| `maxdim` | `:suzuki5` @ dt=0.05 | `:yoshida3` @ dt=0.05 |
|---:|---:|---:|
| 2 | 2.63e-4 | 2.64e-4 (free) |
| 3 | 1.35e-5 | 3.36e-5 (2.5×)  |
| 4 | 1.67e-6 | 3.26e-5 (19×)   |

`maxdim = 2` throws away everything above the 2nd-order content, so the choice of
composition is free there. `maxdim = 4` reproduces the composite and inherits its
constant in full. Note that for a base which is *not* self-adjoint both
compositions give 3rd order anyway (`Ghent2Sym`: local slope 4.0 either way, with
`:yoshida3` ~48× worse), so on those bases `:yoshida3` buys the feasibility and
gives up nothing in order.
"""
function compose_steps(compose::Symbol, dt::Number)
    compose === :none && return (dt,)
    if compose === :suzuki5
        p = 1 / (4 - 4^(1/3))
        return (p*dt, p*dt, (1 - 4p)*dt, p*dt, p*dt)
    elseif compose === :yoshida3
        w = 1 / (2 - 2^(1/3))
        return (w*dt, (1 - 2w)*dt, w*dt)
    end
    throw(ArgumentError("compose must be :none, :suzuki5 or :yoshida3, got $(compose)"))
end

""" Queen of all boilerplate """
function _build_Ut(sites::Vector{<:Index},
    scheme::ExpHRecipe, mp::ModelParams; dt::Number, compose::Symbol)

    steps = compose_steps(compose, dt)
    length(steps) == 1 && return expH(sites, mp, scheme; dt=only(steps))

    # every composition here is palindromic, so only the distinct sub-steps are
    # built and the repeated ones are reused
    built = Tuple{Number,MPO}[]
    factors = map(steps) do s
        k = findfirst(x -> x[1] == s, built)
        k === nothing || return built[k][2]
        U = expH(sites, mp, scheme; dt=s)
        push!(built, (s, U))
        return U
    end

    U = factors[end]
    for k in (length(factors) - 1):-1:1
        U = applyn(factors[k], U)
    end
    return U
end

_compose_kw(build_4o::Bool, compose::Union{Symbol,Nothing}) =
    compose === nothing ? (build_4o ? :suzuki5 : :none) : compose

"""
    build_Ut(sites, scheme, mp; dt, compose=:none)   # or build_4o=true

One step of the evolution operator as an MPO. `compose` selects the symmetric
composition of the 2nd-order gate — see [`compose_steps`](@ref) for the recipes
and for when `:yoshida3` is worth its larger error constant.

`build_4o::Bool` is the older spelling: `build_4o=true` means `compose=:suzuki5`.
An explicit `compose` wins.
"""
function build_Ut(sites::Vector{<:Index}, scheme::ExpHRecipe, mp::ModelParams;
                  dt::Number, build_4o::Bool=false, compose::Union{Symbol,Nothing}=nothing)
    _build_Ut(sites, scheme, mp; dt, compose=_compose_kw(build_4o, compose))
end

function build_Ut(sites::Vector{<:Index}, tp::tMPOParams; dt::Number=tp.dt, kwargs...)
    Ut = build_Ut(sites, tp.scheme, tp.mp; dt, kwargs...)
    return adapt(NDTensors.unwrap_array_type(tp.bl), Ut)
end

function build_Ut(scheme::ExpHRecipe, mp::ModelParams; kwargs...)
    ss = [addtags(sim(mp.phys_site),"Site") for _ in 1:3]
    build_Ut(ss, scheme, mp; kwargs...)
end

function build_Ut(tp::tMPOParams; dt::Number=tp.dt, kwargs...)
    Ut = build_Ut(tp.scheme, tp.mp; dt, kwargs...)
    return adapt(NDTensors.unwrap_array_type(tp.bl), Ut)
end

function build_Ut(b::FwtMPOBlocks; kwargs...)
    build_Ut(b.tp; kwargs...)
end
