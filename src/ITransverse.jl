module ITransverse

using JLD2
using LinearAlgebra
using NDTensors
using ITensors
using ITensorMPS
using ProgressMeter
using ITensors.Adapt
using Statistics: mean, std

using ITensors: OneITensor

import NDTensors: Algorithm, @Algorithm_str

include("ITenUtils/ITenUtils.jl")
using .ITenUtils

include("ChainModels/ChainModels.jl")
using .ChainModels

include("BenchData/BenchData.jl")
using .BenchData

export pMPS,
    overlap_noconj,
    tapply, applyn, applys, applyns,
    isid,
    mergedicts!,
    fidelity, logfidelity,
    normalize_for_overlap!

include("truncation_sweeps/trunc_params.jl")
include("truncation_sweeps/sweeps.jl")
include("truncation_sweeps/sweeps_sym.jl")
include("truncation_sweeps/gen_form_checks.jl")

# from pparams
export TruncParams

export truncate_lsweep, truncate_rsweep, truncate_rsweep!
export truncate_lsweep_sym, truncate_rsweep_sym

export gen_canonical_left, gen_canonical_right

include("entropies/build_entropies.jl")

include("entropies/rdm_svd_entropies.jl")
include("entropies/gen_sym_entropies.jl")
include("entropies/diagonalize_sym_rtm.jl")
include("entropies/compute_rho2.jl")

include("entropies/mutual_infos.jl")

export vn_entanglement_entropy, 
    renyi_entropies,
    gensym_renyi_entropies,
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

export H_ising,
    expH_ising_murg, 
    expH_ising_murg_4o, 

    expH_ising_symm_svd,
    expH_potts_murg, 
    expH_potts_symmetric_svd,
    expH_random_symm_svd_1o

export build_H, build_Ut

export timeEvo_MPO_2ndOrder, timeEvo_MPO_2ndOrder_LRflipped

include("tmpo/construct-tMPO-tMPS.jl")
export construct_tMPS_tMPO

include("tmpo/tmpo_params.jl")
export tMPOParams, ising_tp

include("tmpo/fw_tmpo_blocks.jl")
include("tmpo/fold_tmpo_blocks.jl")

export FoldtMPOBlocks, FwtMPOBlocks

include("tmpo/build_Ut.jl")
include("tmpo/build_ww.jl")

include("tmpo/build_fw_tmpo.jl")
export fw_tMPO, fw_tMPS
include("tmpo/build_fwback_tmpo.jl")
export fwback_tMPO, fwback_tMPS

#from build_tmpo_[fw|expval].jl


include("tmpo/build_fold_tmpo.jl")
include("tmpo/build_fold_tmpo_in.jl")

#from build_fold_tmpo.jl
export 
    folded_tMPO,
    folded_tMPS,
    folded_left_tMPS,
    folded_right_tMPS,
    folded_tMPO_in

include("folding/foldings.jl")


include("columns_envs/columns.jl")

export Columns

include("columns_envs/environments.jl")
include("columns_envs/initialize_envs.jl")
include("columns_envs/update_envs.jl")
include("columns_envs/update_envs_skewed.jl")

export Environments, 
    initialize_envs_rdm, 
    sweep_rebuild_envs_rtm!,
    overlap_envs


include("checkpoints/checkpoints.jl")
export DoCheckpoint

include("power_method/pm_params.jl")

export PMParams

include("power_method/pm.jl")
include("power_method/symm_pm.jl")
#from power_method.jl
export powermethod_op

export powermethod_sym

include("contractions/contract_finite.jl")

include("contractions/expvals_lr.jl")
export expval_LR, compute_expvals

include("lightcone/cone_tmpo.jl")
export folded_tMPO_ext

include("lightcone/cone_params.jl")
export ConeParams

include("lightcone/extend_cone.jl")

include("lightcone/init_cone.jl")

include("lightcone/run_cone.jl")

export init_cone, run_cone

# include("lightcone/cone_envs/cone_columns.jl")
# include("lightcone/cone_envs/extend_cone_envs.jl")
# include("lightcone/cone_envs/shrink_cone.jl")

include("tebd/tebd.jl")

export tebd_ev

end #module ITransverse
