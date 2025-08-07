struct tMPOParams{T<:Union{Float64,ComplexF64}, MP, F}
    dt::T
    expH_func::F
    mp::MP
    nbeta::Int
    bl::ITensor
end

function Base.show(io::IO, tp::tMPOParams)
    println("tMPOParams")
    println("dt:           $(tp.dt)    nbeta : $(tp.nbeta)")
    println("exp(H) func:  $(tp.expH_func)")
    println("Model params: $(tp.mp)")
    println("Init state:   $(array(tp.bl))")
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


""" quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+> """
function ising_tp() 
    return tMPOParams(0.1, build_expH_ising_murg, IsingParams(1.0, 0.4, 0.0), 0, [1.0+0im,1]/sqrt(2))
end

Adapt.adapt_structure(to, x::tMPOParams) = tMPOParams(
    x.dt,
    x.expH_func,  
    x.mp,         
    x.nbeta,
    adapt(to, x.bl)
)
