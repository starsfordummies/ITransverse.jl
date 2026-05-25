module ITransverseCUDAExt

using CUDA
using NDTensors
using ITensors
using ITensors: Algorithm, @Algorithm_str
using ITensorMPS
using Adapt
using ITransverse 

import ITransverse: togpu
import ITransverse: tcontract, trcontract, tlrcontract

NDTensors.cu(x::tMPOParams) = adapt(CuArray, x)

ITransverse.togpu(x) = adapt(CuArray, x)

include("light_cuapply.jl")

end