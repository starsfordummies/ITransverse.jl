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
    SVD_cutoff::Float64
    maxbondim::Int64
    verbose::Bool
    ds2_converged::Float64
    increase_chi::Bool
    plot_s::Bool


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
    return ppm_params(itermax,SVD_cutoff,maxbondim,verbose,ds2_converged, increase_chi, false)
end

#=
function ppm_params(itermax::Int64,
    SVD_cutoff::Float64,
    maxbondim::Int64,
    verbose::Bool,
    ds2_converged::Float64)
    return ppm_params(itermax,SVD_cutoff,maxbondim,verbose,ds2_converged, true, false)
end
=#
