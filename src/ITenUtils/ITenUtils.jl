# ITenUtils — files included directly in ITransverse (no submodule)

# First: `mps_utils.jl` and below already take a `TMPSorMPS`, so the type has to exist by
# then. The operations that need those utilities in turn live in `tmpo/transverse_mps_ops.jl`.
include("transverse_mps.jl")

include("ctruncate.jl")
include("ceigen.jl")

include("utils.jl")

include("matrix_utils.jl")
include("itensor_utils.jl")

include("mps_utils.jl")

include("apply_contract.jl")
include("trunc_apply.jl")

include("custom_svd.jl")

# Symmetric SVD/EIG decompositions
include("svd_sym.jl")
include("eig_sym.jl")

#include("symmsvd_iten.jl")

include("sqrt_itensor.jl")

include("size_estimate.jl")
