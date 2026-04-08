module ITransverseCUDAExt

using CUDA
using ITensors
using ITensorMPS
using NDTensors
using ITensors.Adapt
using ITransverse 

import NDTensors: cu 
import ITransverse.ITenUtils: tocpu, togpu


NDTensors.cu(x::tMPOParams) = tMPOParams(x; bl = NDTensors.cu(x.bl))

ITransverse.ITenUtils.togpu(x) = adapt(CuArray, x) #  NDTensors.cu(x)
ITransverse.ITenUtils.tocpu(x) = adapt(Array, x) #  NDTensors.cpu(x)

end