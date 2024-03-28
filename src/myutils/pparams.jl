# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 


struct pparams
    JXX::Float64
    hz::Float64
    dt::Float64
    nbeta::Int64
    init_state::Vector{ComplexF64}
end

struct ppm_params
    itermax::Int64
    cutoff::Float64
    maxbondim::Int64
    verbose::Bool
    ds2_converged::Float64
    increase_chi::Bool
    plot_s::Bool
    method::String

    function ppm_params(; itermax::Int64 = 400, cutoff::Float64=1e-12, maxbondim::Int64=100, 
        verbose::Bool=false, ds2_converged::Float64=1e-5, increase_chi::Bool=false, plot_s::Bool=false, method::String="SVD")
        return new(itermax, cutoff, maxbondim, verbose, ds2_converged, increase_chi, plot_s, method)
    end
end

struct trunc_params
    cutoff::Float64
    maxbondim::Int64
    method::String
end

 
# past-proof functions to be erased in the future
function ppm_params(itermax::Int64,
    SVD_cutoff::Float64,
    maxbondim::Int64,
    verbose::Bool,
    ds2_converged::Float64,
    increase_chi::Bool=true)
    return ppm_params(itermax,SVD_cutoff,maxbondim,verbose,ds2_converged, increase_chi, false, "SVD")
end


