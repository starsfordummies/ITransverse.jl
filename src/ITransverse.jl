module ITransverse

using JLD2
using LinearAlgebra
using NDTensors
using ITensors
using ITensorMPS
using ProgressMeter
using ITensors.Adapt

include("ITenUtils/ITenUtils.jl")
using .ITenUtils

include("ChainModels/ChainModels.jl")
using .ChainModels


# #from utils.jl
export quick_mps, myrMPO, pMPS,
    overlap_noconj, 
    applyn, applyns, applys,
    isid,
    mergedicts!,
    fidelity

include("sweeps/trunc_params.jl")
include("sweeps/sweeps.jl")
include("sweeps/sweeps_sym.jl")
include("sweeps/gen_form_checks.jl")

# from pparams
export TruncParams

export truncate_lsweep, truncate_rsweep, truncate_rsweep!
export truncate_lsweep_sym, truncate_rsweep_sym

export gen_canonical_left, gen_canonical_right

export check_gencan_left, check_gencan_right


include("entropies/build_entropies.jl")

include("entropies/rdm_svd_entropies.jl")
include("entropies/gen_sym_entropies.jl")
include("entropies/diagonalize_sym_rtm.jl")
include("entropies/compute_rho2.jl")

export vn_entanglement_entropy, 
    renyi_entanglement_entropy!,
    build_entropies,
    diagonalize_rdm,
    generalized_vn_entropy_symmetric,
    generalized_svd_vn_entropy_symmetric,
    generalized_svd_vn_entropy,
    diagonalize_rtm_symmetric,
    gen_renyi2


# from compute_rho2.jl
export rho2, 
    rtm2_contracted

export ModelParams, IsingParams, PottsParams, XXZParams, NoParams

export TruncParams

export build_H_ising,
    build_expH_ising_murg, 
    build_expH_ising_symm_svd,
    build_expH_potts_murg, 
    build_expH_potts_symmetric_svd,
    build_expH_random_symm_svd_1o

export timeEvo_MPO_2ndOrder


include("tmpo/tmpo_params.jl")
export tMPOParams, ising_tp

include("tmpo/build_expH.jl")


include("tmpo/build_ww.jl")

include("tmpo/fw_tmpo_blocks.jl")
include("tmpo/fold_tmpo_blocks.jl")
include("tmpo/generic_tmpo_blocks.jl")

#export rotate_90clockwise
export FoldtMPOBlocks, FwtMPOBlocks

export get_Wc, get_Wl, get_Wr
 #build_WWl, build_WWc, build_WWr, build_WW

include("tmpo/build_tmpo_fw.jl")

#from build_tmpo_[fw|expval].jl
export 
    fw_tMPO, 
    fw_right_tMPS, 
    fw_left_tMPS

include("tmpo/build_fold_tmpo.jl")

#from build_fold_tmpo.jl
export 
     folded_tMPO,
     folded_left_tMPS,
     folded_right_tMPS,
     folded_right_tMPS_in_murg

   #  apply_extend
    
include("power_method/pm_params.jl")

export PMParams

include("power_method/pm.jl")
include("power_method/symm_pm.jl")
#from power_method.jl
export powermethod_op

export powermethod_sym

include("generic/contract_finite.jl")

export contract_finite

include("generic/expvals_lr.jl")
export expval_LR, compute_expvals

include("lightcone/extended_tmpo.jl")
export folded_tMPO_R, folded_tMPO_in

include("lightcone/cone_params.jl")
export ConeParams

include("lightcone/cone.jl")

export init_cone, run_cone

include("tebd/tebd.jl")

export tebd_ev

end #module ITransverse
