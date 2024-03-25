module ITransverse

using LinearAlgebra
using ITensors, NDTensors
using TakagiFactorization
using ProgressMeter


include("myutils/pparams.jl")
include("myutils/utils.jl")
include("myutils/compute_entropies.jl")


# from pparams.jl
export pparams, ppm_params, trunc_params

#from utils.jl
export myrMPS, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor, 
    check_diag_matrix, 
    check_id_matrix,
    isid

#from compute_entropies.jl
export vn_entanglement_entropy_cut, 
    vn_entanglement_entropy, 
    renyi_entanglement_entropy_cut, 
    renyi_entanglement_entropy,
    generalized_entropy_symmetric_cut, 
    generalized_entropy_symmetric, 
    generalized_entropy_cut,
    generalized_entropy


#include("truncations/truncations.jl")
include("truncations/symmetric_svd.jl")
#include("truncations/symm_decompositions.jl")

# TODO include check_equivalence_svd ? 

# from truncations.jl
# export mytruncate, mytruncate_eig, svdtrunc, eigtrunc, eigtrunc, mytrunceig!

#from symmetric_svd.jl
export symmetric_svd_iten, symmetric_svd_ndten, symmetric_svd_takagi_iten,
 symmetric_svd_takagi_arr, symmetric_svd_arr, symmetric_svd_arr,
  symmetric_eig_arr, symmetric_eig_arr

include("truncations/sweeps.jl")
include("truncations/sweeps_sym.jl")

#from sweeps_trunc.jl
export truncate_normalize_sweep, truncate_normalize_sweep_LR

export truncate_normalize_sweep_sym!, truncate_normalize_sweep_sym_right

# from symm_decompositions.jl
export symm_svd, symm_oeig


include("models/ising.jl")
include("models/potts.jl")
include("models/xxzmodel.jl")


# from ising.jl
export build_H_ising_manual,
    build_H_ising_manual_lowtri, 
    build_H_ising, 
    build_H_ising_ZZ_X,
    build_H_ising_YY,
    build_expH_ising_1o, 
    build_expH_ising_2o, 
    build_expH_ising_murg, 
    build_expH_ising_murg_ZZX, 
    build_expH_ising_murg_YY

#from potts.jl
export build_H_potts_manual,
    build_H_potts_manual_lowtri,
    build_H_potts,
    build_expH_potts_2o,
    build_expH_potts_murg

# from xxzmodel.jl
export build_H_XXZ_manual,
    build_H_XXZ_manual_lowtri,
    build_H_XXZ,
    build_expH_XXZ_1o,
    build_expH_XXZ_2o,
    build_expH_XXZ_murg


include("tmpo/build_tmpo.jl")
include("tmpo/build_fold_tmpo.jl")


#from build_tmpo.jl
export build_ising_fw_tMPO,
     build_fw_tMPO, 
     build_potts_fw_tMPO, 
     build_xxmodel_fw_tMPO, 
     build_fw_tMPO, 
     build_fw_tMPO_regul_beta, 
     build_ising_tMPO_regul_beta, 
     build_potts_tMPO_regul_beta, 
     build_xxmodel_tMPO_regul_beta, 
     build_tMPO_regul_beta, 
     build_ising_expval_tMPO, 
     build_expval_tMPO, 
     build_left_tMPS

#from build_fold_tmpo.jl
export build_ising_folded_tMPO,
     build_ising_folded_tMPS,
     build_folded_tMPO, 
     build_potts_folded_tMPO, 
     build_xxmodel_folded_tMPO, 
     build_folded_tMPO, 
     build_folded_tMPO_regul_beta, 
     build_ising_folded_tMPO_regul_beta, 
     build_potts_folded_tMPO_regul_beta, 
     build_xxmodel_folded_tMPO_regul_beta, 
     build_folded_tMPO_regul_beta, 
     build_ising_expval_folded_tMPO, 
     build_expval_folded_tMPO, 
     build_folded_left_tMPS


include("power_method/power_method.jl")

#from power_method.jl
export powermethod, 
    powermethod_sym,
    powermethod_sym_rdm,
    powermethod_fold, 
    powermethod_fold_regul_beta, 
    powermethod_regul_beta


include("lightcone/cone.jl")

export evolve_cone, init_cone_ising

end #module ITransverse