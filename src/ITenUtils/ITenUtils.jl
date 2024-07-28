module ITenUtils

using JLD2
using LinearAlgebra
using ITensors

using NDTensors 

using ITensors.Adapt: adapt

import NDTensors:
 replace_nothing,
 default_use_absolute_cutoff,
 default_use_relative_cutoff,
 expose,
 truncate!!


include("ctruncate.jl")
include("ceigen.jl")

include("utils.jl")

include("matrix_utils.jl")
include("itensor_utils.jl")
include("mps_utils.jl")

# Symmetric SVD/EIG decompositions
include("svd_sym.jl")
include("eig_sym.jl")

#include("symmsvd_iten.jl")

include("bench_data.jl")

include("sqrt_itensor.jl")

include("ext_functions.jl")

include("size_estimate.jl")
include("environments.jl")

export sqrt

export mergedicts!, dictfromlist

#from utils.jl
export quick_mps, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor,
    normbyfactor,
    applyn,
    applyns,
    match_siteinds, match_siteinds!,
    replace_linkinds!,
    phys_ind


# moreutils.jl
export randsymITensor,
    isid,
    pinvten,
    randITensor_decayspec

# matrix_utils.jl
export symmetrize,
    check_id_matrix,
    check_diag_matrix,
    randmat_decayspec

# gen_svdeig_symm.jl
export
    symm_svd,
    symm_oeig,
    mytrunc_svd,
    mytrunc_eig

    
export togpu
export tocpu

end
