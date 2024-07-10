module ITransverseGPUExt

using CUDA
using ITensors
using NDTensors
using ITensors.Adapt
using ITransverse 

Adapt.adapt_structure(to, tp::tmpo_params) = tmpo_params(tp; bl=adapt(to, tp.bl), tr=adapt(to,tp.tr))

function ITransverse.ITenUtils.togpu(x) 
    return NDTensors.cu(x)
end

function ITransverse.ITenUtils.tocpu(x::MPS)
    dtype = mapreduce(NDTensors.unwrap_array_type, promote_type, x)
    if dtype <: CuArray
        return NDTensors.cpu(x)
    end
    return x
end

function ITransverse.ITenUtils.tocpu(x::ITensor)
    dtype = promote_type(NDTensors.unwrap_array_type(x))
    if dtype <: CuArray
        return NDTensors.cpu(x)
    end
    return x
end

end #module ITransverseGPUExt
