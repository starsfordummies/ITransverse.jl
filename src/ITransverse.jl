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

using ITensors.Adapt: adapt

using ITensorMPS: setleftlim!, setrightlim!

using NDTensors:
    replace_nothing,
    default_use_absolute_cutoff,
    default_use_relative_cutoff,
    expose,
    truncate!!

# Collection of utilities 
include("ITenUtils/ITenUtils.jl")

export mergedicts!, mergedicts, dictfromlist

export halfsite

export pMPS,
    overlap_noconj,
    check_symmetry_itensor_mpo,
    check_symmetry_swap,
    normbyfactor,
    ttruncate!,
    tapply, tapplys, applyn, applys, applyns,
    match_siteinds, match_siteinds!,
    replace_linkinds!,
    phys_ind,
    gaugefix_left,
    fidelity, logfidelity, gen_fidelity,
    normalize_for_overlap!,
    allsiteinds,
    tcontract

export randsymITensor,
    isid, isdiag,
    pinvten,
    randITensor_decayspec

export symmetrize,
    check_id_matrix,
    isapproxdiag,
    randmat_decayspec,
    matrix_svd,
    truncated_svd,
    vectorized_identity,
    itensor_to_vector,
    to_itensor,
    vectorized_op,
    trace_mpo, trace_mpo_squared,
    max_diff

export symm_svd, symm_oeig, mytrunc_eig

export beta_lims

include("BenchData/BenchData.jl")
using .BenchData


include("chain_models/model_params.jl")
export ModelParams, IsingParams, PottsParams, XXZParams, NoParams
export TrotterScheme, Murg, SymSVD, Floquet
export expH

include("chain_models/helpers.jl")
include("chain_models/id_mpo.jl")
include("chain_models/ising_parallel.jl")
include("chain_models/potts.jl")
include("chain_models/xxzmodel.jl")
include("chain_models/random_mpo.jl")
include("chain_models/floq_ising.jl")
include("chain_models/trotter_schemes.jl")

export up_state, down_state, plus_state
export vX, vZ, vI


# from ising.jl
export H_ising

#from potts.jl
export H_potts_manual,
    H_potts

export build_H, build_Ut

#include("truncation_sweeps/trunc_params.jl")
include("truncation_sweeps/sweeps.jl")
include("truncation_sweeps/sweeps_sym.jl")
include("truncation_sweeps/gen_orthogonalize.jl")
include("truncation_sweeps/gen_form_checks.jl")
include("truncation_sweeps/trunclr_apply.jl")
include("truncation_sweeps/rtm_r_contract.jl")
include("truncation_sweeps/rtm_lr_contract.jl")

export truncate_sweep, truncate_sweep_rtm
export truncate_lsweep_sym, truncate_rsweep_sym, truncate_sweep_sym

export tlapply, trapply, tlrapply
export TruncLR

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
    generalized_svd_vn_entropy,
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



include("checkpoints/checkpoints.jl")
export DoCheckpoint, write_cp

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

export ConeParams, folded_tMPO_ext, init_cone, run_cone, resume_cone

# include("lightcone/cone_envs/cone_columns.jl")
# include("lightcone/cone_envs/extend_cone_envs.jl")
# include("lightcone/cone_envs/shrink_cone.jl")

include("tebd/tebd.jl")

export tebd_ev

# legacy functions 
include("legacy/old_legacy.jl")

function togpu end


end #module ITransverse
