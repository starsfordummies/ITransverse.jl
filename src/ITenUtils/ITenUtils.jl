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
 truncate!!,
 Algorithm,
 @Algorithm_str

include("ctruncate.jl")
include("ceigen.jl")

include("utils.jl")

include("matrix_utils.jl")
include("itensor_utils.jl")

include("mps_utils.jl")

include("apply_contract.jl")
include("trunc_apply.jl")

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

export TruncatedMPS

#from utils.jl
export pMPS,
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_swap,
    normbyfactor,
    tapply, applyn, applys, applyns,
    match_siteinds, match_siteinds!,
    replace_linkinds!,
    phys_ind,
    gaugefix_left,
    fidelity,
    logfidelity,
    normalize_for_overlap!,
    allsiteinds


# moreutils.jl
export randsymITensor,
    isid,
    isdiag,
    pinvten,
    randITensor_decayspec

# matrix_utils.jl
export symmetrize,
    check_id_matrix,
    isapproxdiag,
    randmat_decayspec,
    matrix_svd,
    vectorized_identity,
    itensor_to_vector,
    to_itensor,
    vectorized_op,
    vectorize_mpo,
    trace_mpo, trace_mpo_squared

# gen_svdeig_symm.jl
export
    symm_svd,
    symm_oeig,
    mytrunc_eig

export beta_lims

export togpu
export tocpu

end
