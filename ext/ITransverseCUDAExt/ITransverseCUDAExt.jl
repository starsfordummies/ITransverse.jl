module ITransverseCUDAExt

using CUDA
using NDTensors
using ITensors
using ITensors: Algorithm
using ITensorMPS
using Adapt
using ITransverse 

import ITransverse: togpu
import ITransverse: tcontract, trcontract, tlrcontract

NDTensors.cu(x::tMPOParams) = tMPOParams(x; bl = NDTensors.cu(x.bl))

ITransverse.togpu(x) = adapt(CuArray, x)

include("light_cuapply.jl")

end