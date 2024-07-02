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
    bl::Vector{ComplexF64}  # bottom -> left(rotated)
    tr::Vector{ComplexF64}  # top -> right(rotated)
end

 # allow for changes on the fly of params
tmpo_params(x::tmpo_params; 
    expH_func=x.expH_func, 
    mp=x.model_params,
    nbeta=x.model,
    bl=x.bl, tr = x.tr) = tmpo_params(expH_func, mp, nbeta, bl, tr)


struct trunc_params
    cutoff::Float64
    maxbondim::Int64
    ortho_method::String

end

trunc_params() = trunc_params(1e-10, 100, "SVD")

 

