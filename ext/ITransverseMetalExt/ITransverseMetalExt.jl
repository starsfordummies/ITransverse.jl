module ITransverseMetalExt

using Metal: MtlArray
using ITensors.Adapt
using ITransverse 

import ITransverse: togpu

ITransverse.togpu(x) = adapt(MtlArray, adapt(ComplexF32, x))

end
