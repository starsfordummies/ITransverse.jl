# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 

""" This is old and should be superseded soon"""
struct pparams
    JXX::Float64
    hz::Float64
    dt::Float64
    nbeta::Int64
    init_state::Vector{ComplexF64}
end

struct model_params
    phys_space::String
    JXX::Float64
    hz::Float64
    λx::Float64
    dt::Float64
end

# allow for changes on the fly of params 
model_params(x::model_params; 
    JXX=x.JXX, 
    hz=x.hz, 
    λx=x.λx, 
    dt=x.dt) = model_params(x.phys_space, JXX, hz, λx, dt)

struct tmpo_params
    time_phys_space::String
    virt_space::String
    expH_func::Function
    mp::model_params
    nbeta::Int64
    init_state::Vector{ComplexF64}
   
end

 # allow for changes on the fly of params
tmpo_params(x::tmpo_params; 
    expH_func=x.expH_func, 
    mp=x.model_params,
    nbeta=x.model,
    init_state=x.init_state) = tmpo_params(x.time_phys_space, x.virt_space, expH_func, mp, nbeta, init_state)


struct ppm_params
    itermax::Int64
    cutoff::Float64
    maxbondim::Int64
    verbose::Bool
    ds2_converged::Float64
    increase_chi::Bool
    plot_s::Bool
    ortho_method::String

    function ppm_params(; itermax::Int64 = 400, cutoff::Float64=1e-12, maxbondim::Int64=100, 
        verbose::Bool=false, ds2_converged::Float64=1e-5, increase_chi::Bool=false, plot_s::Bool=false, ortho_method::String="SVD")
        return new(itermax, cutoff, maxbondim, verbose, ds2_converged, increase_chi, plot_s, ortho_method)
    end


    # past-proof functions to be erased in the future
    function ppm_params(itermax::Int64,
        cutoff::Float64,
        maxbondim::Int64,
        verbose::Bool,
        ds2_converged::Float64,
        increase_chi::Bool=true)
        return new(itermax,cutoff,maxbondim,verbose,ds2_converged, increase_chi, false, "SVD")
    end

end

struct trunc_params
    cutoff::Float64
    maxbondim::Int64
    ortho_method::String
end

 

