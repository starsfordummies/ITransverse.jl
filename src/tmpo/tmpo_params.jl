Base.@kwdef struct tMPOParams{Tdt, Tdbeta, MP, S<:TrotterScheme}
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
tMPOParams(dt::Number, dbeta::Number, scheme::TrotterScheme, mp::ModelParams, nbeta::Int, bl_in::AbstractVector) =
    tMPOParams(dt, dbeta, scheme, mp, nbeta, to_itensor(bl_in, "bl"))

# Short form: omit dbeta (defaults to -im*dt)
tMPOParams(dt::Number, scheme::TrotterScheme, mp::ModelParams, nbeta::Int, bl_in::AbstractVector) =
    tMPOParams(dt, -im*dt, scheme, mp, nbeta, bl_in)

# Allow changes on the fly
tMPOParams(x::tMPOParams; dt=x.dt, dbeta=x.dbeta, scheme=x.scheme, mp=x.mp, nbeta=x.nbeta, bl=x.bl) =
    tMPOParams(; dt, dbeta, scheme, mp, nbeta, bl)


function tMPOParams(x::Nothing; bl)
    blt = to_itensor(bl, "bl")
    return tMPOParams(; dt=NaN, dbeta=nothing, scheme=Murg(), mp=NoParams(Index(dim(blt))), bl=blt)
end

""" Quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+> """
function ising_tp(;hz = 0.4, integrable::Bool=true, init_state=[1,0]) 
    tp = if integrable
         tMPOParams(0.1, -0.1im, Murg(), IsingParams(1.0, hz, 0.0),   0, init_state)
    else
        tMPOParams(0.1, -0.1im, Murg(), IsingParams(1.0, -1.05, 0.5), 0, init_state)
    end

    return tp
end


Adapt.adapt_structure(to, x::tMPOParams) = tMPOParams(x; bl=adapt(to, x.bl))
