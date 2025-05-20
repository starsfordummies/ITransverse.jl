struct tMPOParams{T<:Union{Float64,ComplexF64}, MP, F}
    dt::T
    expH_func::F
    mp::MP
    nbeta::Int
    bl::ITensor
    tr::ITensor
end


function build_default_tr(mp::ModelParams)
    close_id = op("I", mp.phys_site)  # Make identity op
    return close_id * combiner(inds(close_id), tags="tr")
end

# Master constructor
function tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int,
    bl_in
)
    blt = to_itensor(bl_in, "bl")

    # handle tr cases
    trt = build_default_tr(mp)
  
    return tMPOParams(dt, expH_func, mp, nbeta, blt, trt)
end

function tMPOParams(
    dt::Number,
    expH_func::Function,
    mp::ModelParams,
    nbeta::Int,
    bl_in::Union{Vector,ITensor},
    tr_in::Vector
)
    blt = to_itensor(bl_in, "bl")
    trt = to_itensor(tr_in, "tr")

    return tMPOParams(dt, expH_func, mp, nbeta, blt, trt)
end


 # allow for changes on the fly of params
tMPOParams(x::tMPOParams; 
    dt = x.dt,
    expH_func = x.expH_func, 
    mp = x.mp,
    nbeta = x.nbeta,
    bl = x.bl, 
    tr = x.tr) = tMPOParams(dt, expH_func, mp, nbeta, bl, tr)


""" quick defaults for parallel field Ising, for playing around: J=1 hz=0.4, gx=0, init_state= |+> """
ising_tp() = tMPOParams(0.1, build_expH_ising_murg, 
    IsingParams(1.0, 0.4, 0.0), 0, [1.0+0im,1]/sqrt(2), [1,0,0,1])