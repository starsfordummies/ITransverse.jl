module ITenUtils

using JLD2
using LinearAlgebra

using NDTensors 

using ITensors
using ITensorMPS

using ProgressMeter

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
include("apply_contract.jl")

# Symmetric SVD/EIG decompositions
include("svd_sym.jl")
include("eig_sym.jl")

#include("symmsvd_iten.jl")

include("sqrt_itensor.jl")

include("ext_functions.jl")

include("size_estimate.jl")

include("vectorize_mpo.jl")


export sqrt

export mergedicts!, mergedicts, dictfromlist

#from utils.jl
export quick_mps, quick_mpo, pMPS,
    quick_psipsio, myrMPO,
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_swap,
    normbyfactor,
    applyn,
    applyns,
    applys,
    match_siteinds, match_siteinds!,
    replace_linkinds!,
    phys_ind,
    gaugefix_left,
    fidelity


# moreutils.jl
export randsymITensor,
    isid,
    isdiag,
    pinvten,
    randITensor_decayspec

# matrix_utils.jl
export symmetrize,
    check_id_matrix,
    check_diag_matrix,
    randmat_decayspec,
    matrix_svd,
    vectorized_identity,
    itensor_to_vector,
    to_itensor,
    vectorized_op

# gen_svdeig_symm.jl
export
    symm_svd,
    symm_oeig,
    mytrunc_svd,
    mytrunc_eig

export beta_lims

export togpu
export tocpu

end
