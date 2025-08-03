module ITransverseMetalExt

using Metal
using ITensors
using ITensorMPS
using NDTensors
using ITensors.Adapt
using ITransverse 

# allow printing of arrays on GPU even if sloow 
function Base.show(io::IO, arr::Metal.AbstractGPUArray)
    Metal.@allowscalar begin
        invoke(Base.show, Tuple{IO, typeof(arr)}, io, arr)
    end
end

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
