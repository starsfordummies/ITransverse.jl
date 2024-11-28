struct tMPOParams{T <:Union{Float64,ComplexF64}}
    dt::T
    expH_func::Function
    mp::ModelParams
    nbeta::Int64
    bl::ITensor  # bottom -> left(rotated)
    tr::ITensor  # top -> right(rotated)

end

function tMPOParams(dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::Vector{<:Number},  # bottom -> left(rotated)
    tr::Vector{<:Number}) 

    
    work_type = ComplexF64

    blt = adapt(work_type,ITensor(bl, Index(length(bl), "bl")))
    trt = adapt(work_type,ITensor(tr, Index(length(tr), "tr")))
    return tMPOParams(dt, expH_func, mp, nbeta, blt, trt)
end

function tMPOParams(dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::ITensor,  # bottom -> left(rotated)
    tr::Vector{<:Number}) 

    work_type = ComplexF64

    trt = adapt(work_type,ITensor(tr, Index(length(tr), "tr")))
    return tMPOParams(dt, expH_func, mp, nbeta, bl, trt)
end

tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::Vector{<:Number}) = tMPOParams(dt, expH_func, mp, nbeta, bl, [1,0,0,1])

tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int64,
    bl::ITensor) = tMPOParams(dt, expH_func, mp, nbeta, bl,  ITensor([1,0,0,1], Index(length(tr), "tr")))


 # allow for changes on the fly of params
tMPOParams(x::tMPOParams; 
    dt::Number = x.dt,
    expH_func::Function=x.expH_func, 
    mp::ModelParams=x.mp,
    nbeta::Int64=x.nbeta,
    bl=x.bl, 
    tr = x.tr) = tMPOParams(dt, expH_func, mp, nbeta, bl, tr)


# quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+>
ising_tp() = tMPOParams(0.1, build_expH_ising_murg, 
    ModelParams("S=1/2", 1.0, 0.4, 0.0), 0, [1.0+0im,1]/sqrt(2), [1,0,0,1])