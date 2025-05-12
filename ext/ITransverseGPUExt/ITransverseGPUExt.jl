module ITransverseGPUExt

using CUDA
using ITensors
using ITensorMPS
using NDTensors
using ITensors.Adapt
using ITransverse 


Adapt.adapt_structure(to, x::tMPOParams) = tMPOParams(
    dt,
    expH_func,  # This is safe even for functions (identity)
    x.mp,          # assumes ModelParams is also adapt-compatible
    x.nbeta,
    adapt(to, x.bl),
    adapt(to, x.tr)
)


Adapt.adapt_structure(to, b::FwtMPOBlocks) = FwtMPOBlocks(
    adapt(to, b.Wl), 
    adapt(to, b.Wc), 
    adapt(to, b.Wr),
    adapt(to, b.tp), 
    b.rot_inds
)

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
