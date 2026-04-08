module ITransverseMetalExt

using Metal: MtlArray
using ITensors.Adapt
using ITransverse 

import ITransverse.ITenUtils: tocpu, togpu

ITransverse.ITenUtils.togpu(x) = adapt(MtlArray, x)

end
