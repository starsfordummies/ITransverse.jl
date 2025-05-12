struct tMPOParams{T<:Union{Float64,ComplexF64}, MP, F}
    dt::T
    expH_func::F
    mp::MP
    nbeta::Int
    bl::ITensor
    tr::ITensor
end

function tMPOParams(dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::ITensor,  # bottom -> left(rotated)
    tr::Vector{<:Number})

    trt = ITensor(ComplexF64.(tr), Index(length(tr), "tr"))
    return tMPOParams(dt, expH_func, mp, nbeta, bl, trt)
end

function tMPOParams(dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::Vector{<:Number},
    tr::Vector{<:Number})

    blt = ITensor(ComplexF64.(bl), Index(length(bl), "bl"))
    trt = ITensor(ComplexF64.(tr), Index(length(tr), "tr"))
    return tMPOParams(dt, expH_func, mp, nbeta, blt, trt)
end

tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::ITensor) = tMPOParams(dt, expH_func, mp, nbeta, bl,  ITensor(ComplexF64.([1,0,0,1]), Index(4, "tr")))


 # allow for changes on the fly of params
tMPOParams(x::tMPOParams; 
    dt = x.dt,
    expH_func = x.expH_func, 
    mp = x.mp,
    nbeta = x.nbeta,
    bl = x.bl, 
    tr = x.tr) = tMPOParams(dt, expH_func, mp, nbeta, bl, tr)


# quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+>
ising_tp() = tMPOParams(0.1, build_expH_ising_murg, 
    IsingParams(1.0, 0.4, 0.0), 0, [1.0+0im,1]/sqrt(2), [1,0,0,1])