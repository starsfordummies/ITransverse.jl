
struct tmpo_params
    expH_func::Function
    mp::model_params
    nbeta::Int64
    bl::ITensor  # bottom -> left(rotated)
    tr::ITensor  # top -> right(rotated)

    tmpo_params(expH_func::Function,
    mp::model_params,
    nbeta::Int64,
    bl::ITensor,  # bottom -> left(rotated)
    tr::ITensor) = new(expH_func, mp, nbeta, bl, tr)
end

function tmpo_params(expH_func::Function,
    mp::model_params,
    nbeta::Int64,
    bl::Vector{<:Number},  # bottom -> left(rotated)
    tr::Vector{<:Number}) 

    blt = ITensor(bl, Index(length(bl), "bl"))
    trt = ITensor(tr, Index(length(tr), "tr"))
    return tmpo_params(expH_func, mp, nbeta, blt, trt)
end

tmpo_params(
    expH_func::Function,
    mp::model_params,
    nbeta::Int64,
    bl::Vector{<:Number}) = tmpo_params(expH_func, mp, nbeta, bl, [1,0,0,1])

tmpo_params(
    expH_func::Function,
    mp::model_params,
    nbeta::Int64,
    bl::ITensor) = tmpo_params(expH_func, mp, nbeta, bl,  ITensor([1,0,0,1], Index(length(tr), "tr")))

 # allow for changes on the fly of params
tmpo_params(x::tmpo_params; 
    expH_func::Function=x.expH_func, 
    mp::model_params=x.mp,
    nbeta::Int64=x.nbeta,
    bl::ITensor=x.bl, tr::Vector{<:Number} = x.tr) = tmpo_params(expH_func, mp, nbeta, bl, tr)


# quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+>
ising_tp() = tmpo_params(build_expH_ising_murg, 
    model_params("S=1/2", 1.0, 0.4, 0.0, 0.1), 0, [1/sqrt(2),1/sqrt(2)], [1,0,0,1])