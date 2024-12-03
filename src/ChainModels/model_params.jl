# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 

struct ModelParams
    phys_space::String
    JXX::Float64
    hz::Float64
    λx::Float64

    ModelParams(phys_space, JXX, hz, λx) = new(phys_space, JXX, hz, λx)
end

# allow for changes on the fly of params 
ModelParams(x::ModelParams; 
    JXX=x.JXX, 
    hz=x.hz, 
    λx=x.λx) = ModelParams(x.phys_space, JXX, hz, λx)

