module ITransverseGPUExt

using CUDA

using ITensors
using ITensorMPS
using ITransverse 
using ProgressMeter
using JLD2


# include("gpu_expvals.jl")
# include("gpu_sweeps.jl")
# include("gpu_cone.jl")
# include("gpu_cone_svd.jl")


togpu(x) = NDTensors.cu(x)

# function device(b::Struct)
#     cuvals= NDTensors.cu.(getfield.((b,), fieldnames(typeof(b))))
#     return typeof(b)(cuvals)
# end

# function device(b::FoldtMPOBlocks)
#     return FoldtMPOBlocks(device(b.WWl), device(b.WWc), device(b.WWr), device(b.rho0), tp, inds_ww)
# end

    
end #module ITransverseGPUExt
