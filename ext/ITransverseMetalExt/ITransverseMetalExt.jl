module ITransverseMetalExt

using Metal
using ITensors
using ITensorMPS
using NDTensors
using ITensors.Adapt
using ITransverse 

function ITransverse.ITenUtils.togpu(x) 
    return mtl(x)
end

function ITransverse.ITenUtils.tocpu(x::ITensorMPS.MPS)
    dtype = mapreduce(NDTensors.unwrap_array_type, promote_type, x)
    if dtype <: MtlArray
        return NDTensors.cpu(x)
    end
    return x
end

function ITransverse.ITenUtils.tocpu(x::ITensor)
    dtype = promote_type(NDTensors.unwrap_array_type(x))
    if dtype <: MtlArray
        return NDTensors.cpu(x)
    end
    return x
end

end
