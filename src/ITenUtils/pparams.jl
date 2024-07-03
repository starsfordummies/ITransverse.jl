# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 

struct model_params
    phys_space::String
    JXX::Float64
    hz::Float64
    λx::Float64
    dt::ComplexF64
end

# allow for changes on the fly of params 
model_params(x::model_params; 
    JXX=x.JXX, 
    hz=x.hz, 
    λx=x.λx, 
    dt=x.dt) = model_params(x.phys_space, JXX, hz, λx, dt)



struct tmpo_params
    expH_func::Function
    mp::model_params
    nbeta::Int64
    bl::Vector{<:Number}  # bottom -> left(rotated)
    tr::Vector{<:Number}  # top -> right(rotated)

end

tmpo_params(
    expH_func::Function,
    mp::model_params,
    nbeta::Int64,
    bl::Vector{<:Number}) = tmpo_params(expH_func, mp, nbeta, bl, [1,0,0,1])

 # allow for changes on the fly of params
tmpo_params(x::tmpo_params; 
    expH_func::Function=x.expH_func, 
    mp::model_params=x.mp,
    nbeta::Int64=x.nbeta,
    bl::Vector{<:Number}=x.bl, tr::Vector{<:Number} = x.tr) = tmpo_params(expH_func, mp, nbeta, bl, tr)



struct trunc_params
    cutoff::Float64
    maxbondim::Int64
    ortho_method::String

end

trunc_params() = trunc_params(1e-10, 100, "SVD")

 

