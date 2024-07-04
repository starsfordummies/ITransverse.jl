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

