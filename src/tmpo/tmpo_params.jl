struct tMPOParams{T<:Union{Float64,ComplexF64}, MP, F}
    dt::T
    expH_func::F
    mp::MP
    nbeta::Int
    bl::ITensor
end

function Base.show(io::IO, tp::tMPOParams)
    println(io, "tMPOParams")
    println(io, "dt:           $(tp.dt)    nbeta : $(tp.nbeta)")
    println(io, "exp(H) func:  $(tp.expH_func)")
    println(io, "Model params: $(tp.mp)")
    println(io, "Init state:   $(array(tp.bl))")
end


# Master constructor
function tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int,
    bl_in::AbstractVector
)
    blt = to_itensor(bl_in, "bl")

    return tMPOParams(dt, expH_func, mp, nbeta, blt)
end


 # allow for changes on the fly of params
tMPOParams(x::tMPOParams; 
    dt = x.dt,
    expH_func = x.expH_func, 
    mp = x.mp,
    nbeta = x.nbeta,
    bl = x.bl) = tMPOParams(dt, expH_func, mp, nbeta, bl)


function tMPOParams(x::Nothing; bl)
    blt = to_itensor(bl, "bl")
    phys_site = Index(dim(blt))
    mp = NoParams(phys_site)
    return tMPOParams(NaN, nothing, mp, 0, blt)
end

""" Quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+> """
function ising_tp(;hz = 0.4, integrable::Bool=true) 
    tp = if integrable
         tMPOParams(0.1, build_expH_ising_murg, IsingParams(1.0, hz, 0.0),   0, [1.0+0im,1]/sqrt(2))
    else
        tMPOParams(0.1,  build_expH_ising_murg, IsingParams(1.0, -1.05, 0.5), 0, [1,0])
    end

    return tp
end


Adapt.adapt_structure(to, x::tMPOParams) = tMPOParams(
    x.dt,
    x.expH_func,  
    x.mp,         
    x.nbeta,
    adapt(to, x.bl)
)
