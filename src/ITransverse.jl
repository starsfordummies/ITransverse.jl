module ITransverse

using JLD2
using LinearAlgebra
using NDTensors
using ITensors
using ITensorMPS
using ProgressMeter


include("ITenUtils/ITenUtils.jl")
using .ITenUtils

export sqrt 

# #from utils.jl
export quick_mps, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor,
    applys,
    isid,
    mergedicts!,
    merge_siteinds!
    # check_diag_matrix, 
    # check_id_matrix,
    # isid


include("sweeps/sweeps.jl")
include("sweeps/alt_sweeps.jl")
include("sweeps/sweeps_sym.jl")
include("sweeps/gen_form_checks.jl")

export truncate_lsweep, truncate_rsweep
export truncate_lsweep_sym, truncate_rsweep_sym

#from sweeps.jl
export truncate_normalize_sweep, 
    truncate_normalize_sweep_LR

# from alt_sweeps.jl
export truncate_sweep_keep_lenv, truncate_sweep_aggressive_normalize
# sweeps_sym.jl
export  truncate_normalize_sweep_sym,
    truncate_normalize_sweep_sym!,
    truncate_normalize_sweep_sym_right,
    gen_canonical_left, gen_canonical_right



include("entropies/rdm_svd_entropies.jl")
include("entropies/gen_entropies.jl")
include("entropies/gen_sym_entropies.jl")
include("entropies/diagonalize_sym_rtm.jl")
include("entropies/compute_rho2.jl")
#include("entropies/ent_segment.jl")


#from compute_entropies.jl
export vn_entanglement_entropy, 
    renyi_entanglement_entropy!,
    generalized_entropy_symmetric_cut, 
    generalized_entropy_symmetric, 
    #generalized_entropy_cut,
    generalized_entropy,
    generalized_renyi_entropy,
    build_entropies

# from gen_form_checks
export check_gencan_left, check_gencan_right

# from compute_rho2.jl
export rho2, 
    rtm2_contracted, 
    rtm2_bruteforce,
    diagonalize_rtm_right_gen_sym,
    diagonalize_rtm_left_gen_sym

 
include("ChainModels/ChainModels.jl")
using .ChainModels

export ising_tp

export trunc_params, 
    model_params, 
    tmpo_params

export build_expH_ising_murg, 
build_expH_potts_murg, 
build_expH_potts_symmetric_svd,
build_expH_ising_parallel_field_murg

include("tmpo/build_ww.jl")

export rotate_90clockwise, FoldtMPOBlocks, build_WWl, build_WWc, build_WWr, build_WW

include("tmpo/build_tmpo_fw.jl")

#from build_tmpo_[fw|expval].jl
export
    fw_tMPO,
    #build_ising_fw_tMPO_regul_beta, 
    #build_ising_fw_tMPO,
    #build_potts_fw_tMPO, 
    #build_xxmodel_fw_tMPO, 
    build_potts_fw_tMPO_regul_beta, 
    build_xxmodel_fw_tMPO_regul_beta, 
    #build_tMPO_regul_beta, 
    build_expval_tMPO, 
    build_ising_expval_tMPO
    #build_left_tMPS


include("tmpo/build_fold_tmpo.jl")
#include("tmpo/ising_fold_tmpo.jl")

#from build_fold_tmpo.jl
export 
     #build_ising_folded_tMPO,  # superseded by build_ham_ ? 
     #build_ising_folded_tMPS,
    #  build_folded_tMPO_regul_beta, 
    #  build_folded_left_tMPS,
    #  folded_tMPO,
    #  folded_open_tMPO

     folded_tMPO,
     folded_tMPO_R,
     folded_right_tMPS,
     apply_extend
    
include("power_method/pm_params.jl")

export PMParams

include("power_method/pm.jl")
include("power_method/symm_pm.jl")

#from power_method.jl
export powermethod, 
    powermethod_converge_norm,
    powermethod_converge_eig,
    powermethod_fold,
    powermethod_Ronly,
    pm_all,
    pm_svd


export powermethod_sym


include("lightcone/expvals_lr.jl")
export expval_LR, compute_expvals

include("lightcone/cone.jl")

export init_cone, run_cone

include("ext_functions.jl")

export gpu_run_cone, 
gpu_run_cone_svd,
gpu_truncate_sweep,
gpu_truncate_sweep!,
gpu_expval_LR,
cpu_expval_LR,
gpu_expval_LL_sym,
gpu_compute_expvals

export plotr,ploti,plotri 


end #module ITransverse