module ITransverseCUDAExt

using CUDA
using NDTensors
using ITensors.Adapt
using ITransverse 

import ITransverse: togpu
import ITransverse: tcontract
include("light_cuapply.jl")

NDTensors.cu(x::tMPOParams) = tMPOParams(x; bl = NDTensors.cu(x.bl))

ITransverse.togpu(x) = adapt(CuArray, x)

end