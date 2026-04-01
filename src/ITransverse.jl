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

using ITensors: @Algorithm_str, Algorithm

include("ITenUtils/ITenUtils.jl")
using .ITenUtils

export pMPS,
    overlap_noconj,
    tapply, applyn, applys, applyns,
    isid,
    mergedicts!,
    fidelity, logfidelity,
    gen_fidelity,
    normalize_for_overlap!


include("BenchData/BenchData.jl")
using .BenchData


include("chain_models/model_params.jl")
export ModelParams, IsingParams, PottsParams, XXZParams, NoParams

include("chain_models/helpers.jl")
include("chain_models/id_mpo.jl")
include("chain_models/ising_parallel.jl")
include("chain_models/potts.jl")
include("chain_models/xxzmodel.jl")
include("chain_models/random_mpo.jl")
include("chain_models/floq_ising.jl")

export up_state, down_state, plus_state
export vX, vZ, vI


# from ising.jl
export H_ising, 
    expH_ising_murg,
    expH_ising_symm_svd,
    expH_ising_murg_4o

#from potts.jl
export H_potts_manual,
    H_potts,
    #expH_potts_2o,
    expH_potts_murg,
    expH_potts_symmetric_svd

export expH_random_symm_svd_1o

export build_H, build_Ut
#export timeEvo_MPO_2ndOrder, timeEvo_MPO_2ndOrder_LRflipped


include("truncation_sweeps/trunc_params.jl")
include("truncation_sweeps/sweeps.jl")
include("truncation_sweeps/sweeps_sym.jl")
include("truncation_sweeps/gen_orthogonalize.jl")
include("truncation_sweeps/gen_form_checks.jl")
include("truncation_sweeps/tlrcontract.jl")

export TruncParams

export truncate_sweep, truncate_sweep_rtm
export truncate_lsweep_sym, truncate_rsweep_sym, truncate_sweep_sym

export tlapply, trapply, tlrapply

export gen_canonical

include("entropies/build_entropies.jl")

include("entropies/rdm_svd_entropies.jl")
include("entropies/gen_sym_entropies.jl")
include("entropies/diagonalize_sym_rtm.jl")
include("entropies/compute_rho2.jl")

include("entropies/mutual_infos.jl")
include("entropies/fwback_ents.jl")

export vn_entanglement_entropy, 
    renyi_entropies,
    gensym_renyi_entropies,
    diagonalize_rdm,
    generalized_vn_entropy_symmetric,
    diagonalize_rtm_symmetric,
    gen_renyi2


# from compute_rho2.jl
export rho2, rtm2_contracted

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


include("tmpo/build_fold_tmpo.jl")
include("tmpo/build_fold_tmpo_in.jl")

export 
    folded_tMPO,
    folded_tMPS,
    folded_left_tMPS,
    folded_right_tMPS,
    folded_tMPO_in

include("folding/foldings.jl")
include("folding/vectorize_mpo.jl")


include("columns_envs/columns.jl")
include("columns_envs/environments.jl")
include("columns_envs/initialize_envs.jl")
include("columns_envs/sweeps_envs.jl")

export Columns, Environments, 
    initialize_envs, 
    sweep_rebuild_envs_rtm!, sweep_rebuild_envs_rtm_twocol!,
    overlap_envs


include("checkpoints/checkpoints.jl")
export DoCheckpoint

include("power_method/pm_params.jl")
include("power_method/pm.jl")
include("power_method/symm_pm.jl")
export PMParams, powermethod_op, powermethod_sym

include("contractions/contract_finite.jl")

include("contractions/expvals_lr.jl")
export expval_LR, compute_expvals

include("lightcone/cone_tmpo.jl")
include("lightcone/cone_params.jl")
include("lightcone/init_cone.jl")
include("lightcone/run_cone.jl")

export ConeParams, folded_tMPO_ext, init_cone, run_cone

# include("lightcone/cone_envs/cone_columns.jl")
# include("lightcone/cone_envs/extend_cone_envs.jl")
# include("lightcone/cone_envs/shrink_cone.jl")

include("tebd/tebd.jl")

export tebd_ev

end #module ITransverse
