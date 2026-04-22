module ITransverseMetalExt

using Metal: MtlArray
using ITensors: Algorithm
using Adapt
using ITensors, ITensorMPS 
using ITransverse 

import ITransverse: togpu

import ITransverse: tcontract
ITransverse.togpu(x) = adapt(MtlArray, adapt(ComplexF32, x))

function ITransverse.tcontract(::Algorithm{:mtldensitymatrix},
        A::MPO,
        ψ::MPS)

        return ψ
end

end
