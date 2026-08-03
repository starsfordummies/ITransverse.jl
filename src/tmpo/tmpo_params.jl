mutable struct tMPOParams{Tdt<:Number, Tdbeta, MP<:ModelParams, S<:ExpHRecipe}
    dt::Tdt
    dbeta::Tdbeta
    mp::MP
    scheme::S
    nbeta::Int
    bl::ITensor
end

function tMPOParams(mp::ModelParams; 
    dt=0.1, 
    dbeta=-im*dt,
    scheme=default_scheme(mp), 
    nbeta=0, 
    init_state)
    tMPOParams(dt, dbeta, mp, scheme, nbeta, to_boundary(init_state))
end

function Base.show(io::IO, tp::tMPOParams)
    println(io, "tMPOParams:   dt: $(tp.dt) | dbeta: $(tp.dbeta)  || nbeta : $(tp.nbeta)")
    println(io, "scheme:       $(tp.scheme)  |  Model params: $(tp.mp)")
    if is_product_boundary(tp.bl)
        println(io, "Init state:   $(array(tp.bl))")
    else
        println(io, "Init state:   non-product, χ=$(boundary_linkdim(tp.bl)), inds $(inds(tp.bl))")
    end
end



function tMPOParams(x::Nothing; bl)
    blt = to_boundary(bl)
    return tMPOParams(NoParams(Index(dim(blt))); dt=NaN, dbeta=nothing, scheme=Murg(), init_state=blt)
end

""" Quick defaults for parallel field Ising (kept for backward compatibility). """
ising_tp(; hz=0.4, integrable::Bool=true, init_state=[1,0]) =
    tMPOParams(integrable ? IsingParams(1.0, hz, 0.0) : IsingParams(1.0, -1.05, 0.5); init_state)


Adapt.adapt_structure(to, x::tMPOParams) =
    tMPOParams(x.mp; dt=x.dt, dbeta=x.dbeta, scheme=x.scheme, nbeta=x.nbeta, init_state=adapt(to, x.bl))
