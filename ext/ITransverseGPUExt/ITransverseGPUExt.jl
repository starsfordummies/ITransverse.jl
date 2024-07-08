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


function ITransverse.togpu(x) 
    return NDTensors.cu(x)
end

function ITransverse.togpu(b::tmpo_params)
     bl_gpu = NDTensors.cu(b.bl)
     tr_gpu = NDTensors.cu(b.tr)

     return tmpo_params(b; bl = bl_gpu, tr = tr_gpu)
end


function ITransverse.tocpu(x)
    dtype = promote_type(NDTensors.unwrap_array_type(x))
    if dtype <: CuArray
        return NDTensors.cpu(x)
    end
    return x
end

# function ITransverse.togpu(x) 
#     return NDTensors.cpu(x)
# end

# function device(b::Struct)
#     cuvals= NDTensors.cu.(getfield.((b,), fieldnames(typeof(b))))
#     return typeof(b)(cuvals)
# end

# function device(b::FoldtMPOBlocks)
#     return FoldtMPOBlocks(device(b.WWl), device(b.WWc), device(b.WWr), device(b.rho0), tp, inds_ww)
# end

# export ITransverse.togpu
    
end #module ITransverseGPUExt
