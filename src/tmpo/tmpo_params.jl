Base.@kwdef struct tMPOParams{Tdt, Tdbeta, MP, S<:ExpHRecipe}
    dt::Tdt
    dbeta::Tdbeta = dt
    scheme::S
    mp::MP
    nbeta::Int = 0
    bl::ITensor
end

function Base.show(io::IO, tp::tMPOParams)
    println(io, "tMPOParams:   dt: $(tp.dt) | dbeta: $(tp.dbeta)  || nbeta : $(tp.nbeta)")
    println(io, "scheme:       $(tp.scheme)  |  Model params: $(tp.mp)")
    println(io, "Init state:   $(array(tp.bl))")
end


# Master constructor: accepts AbstractVector for bl
tMPOParams(dt::Number, dbeta::Number, scheme::ExpHRecipe, mp::ModelParams, nbeta::Int, bl_in::AbstractVector) =
    tMPOParams(dt, dbeta, scheme, mp, nbeta, to_itensor(bl_in, "bl"))

# Short form: omit dbeta (defaults to -im*dt)
tMPOParams(dt::Number, scheme::ExpHRecipe, mp::ModelParams, nbeta::Int, bl_in::AbstractVector) =
    tMPOParams(dt, -im*dt, scheme, mp, nbeta, bl_in)

# Allow changes on the fly
tMPOParams(x::tMPOParams; dt=x.dt, dbeta=x.dbeta, scheme=x.scheme, mp=x.mp, nbeta=x.nbeta, bl=x.bl) =
    tMPOParams(; dt, dbeta, scheme, mp, nbeta, bl)


function tMPOParams(x::Nothing; bl)
    blt = to_itensor(bl, "bl")
    return tMPOParams(; dt=NaN, dbeta=nothing, scheme=Murg(), mp=NoParams(Index(dim(blt))), bl=blt)
end

""" Build tMPOParams from a ModelParams with sensible per-model defaults.
Keyword arguments override any default: `dt`, `nbeta`, `scheme`, `init_state`.
"""
function tMPOParams(mp::ModelParams;
        dt::Number         = 0.1,
        nbeta::Int         = 0,
        scheme::ExpHRecipe = default_scheme(mp),
        init_state         = default_bl(mp))
    tMPOParams(dt, scheme, mp, nbeta, init_state)
end

""" Quick defaults for parallel field Ising (kept for backward compatibility). """
ising_tp(; hz=0.4, integrable::Bool=true, init_state=[1,0]) =
    tMPOParams(integrable ? IsingParams(1.0, hz, 0.0) : IsingParams(1.0, -1.05, 0.5);
               init_state)


Adapt.adapt_structure(to, x::tMPOParams) = tMPOParams(x; bl=adapt(to, x.bl))
