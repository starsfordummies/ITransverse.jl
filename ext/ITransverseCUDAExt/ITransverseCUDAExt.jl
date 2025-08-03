module ITransverseCUDAExt

using CUDA
using ITensors
using ITensorMPS
using NDTensors
using ITensors.Adapt
using ITransverse 

function Base.show(io::IO, arr::CUDA.AbstractGPUArray)
    CUDA.@allowscalar begin
        invoke(Base.show, Tuple{IO, typeof(arr)}, io, arr)
    end
end

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

end
