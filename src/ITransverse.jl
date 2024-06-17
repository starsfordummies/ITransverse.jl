module ITransverse

using JLD2
using LinearAlgebra
using NDTensors
using ITensors, ITensorMPS
using ProgressMeter


# pkg_dir() = joinpath(dirname(pathof(@__MODULE__)), "..")

# function _parse_project_toml(field::String)
#     return Pkg.TOML.parsefile(joinpath(pkg_dir(), "Project.toml"))[field]
# end

# version() = VersionNumber(_parse_project_toml("version"))

include("ITenUtils/ITenUtils.jl")
using .ITenUtils

include("IGensors/src/IGensors.jl")
using .IGensors

# # from pparams.jl
export pparams, 
    ppm_params, 
    trunc_params, 
    model_params, 
    tmpo_params

# #from utils.jl
export myrMPS, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor
    # check_diag_matrix, 
    # check_id_matrix,
    # isid

# #from compute_entropies.jl


include("truncations/sweeps.jl")
include("truncations/alt_sweeps.jl")
include("truncations/sweeps_sym.jl")

#from sweeps.jl
export truncate_normalize_sweep, 
    truncate_normalize_sweep_LR

# from alt_sweeps.jl
export truncate_sweep_keep_lenv, truncate_sweep_aggressive_normalize
# sweeps_sym.jl
export  truncate_normalize_sweep_sym,
    truncate_normalize_sweep_sym!,
    truncate_normalize_sweep_sym_right,
    gen_canonical_left


include("entropies/gen_form_checks.jl")

include("entropies/entropies.jl")
include("entropies/gen_entropies.jl")
include("entropies/compute_rho2.jl")
#include("entropies/ent_segment.jl")


#from compute_entropies.jl
export vn_entanglement_entropy_cut, 
    vn_entanglement_entropy, 
    renyi_entanglement_entropy_cut, 
    renyi_entanglement_entropy,
    generalized_entropy_symmetric_cut, 
    generalized_entropy_symmetric, 
    #generalized_entropy_cut,
    generalized_entropy,
    generalized_renyi_entropy,
    build_entropies

# from gen_form_checks
export check_gencan_left_phipsi

# from compute_rho2.jl
export rho2, 
    rtm2_contracted, 
    rtm2_bruteforce,
    #gen_symm_diagonalize_rtm,
    diagonalize_rtm_sym_gauged

 



# from symm_decompositions.jl
#export symm_svd, symm_oeig


# include("models/ising.jl")
# include("models/potts.jl")
# include("models/xxzmodel.jl")

include("ChainModels/ChainModels.jl")
using .ChainModels

export pparams, ppm_params, trunc_params
export build_expH_ising_murg, 
build_expH_potts_murg, 
build_expH_potts_symmetric_svd,
build_expH_ising_parallel_field_murg



include("tmpo/build_tmpo_fw.jl")
include("tmpo/build_tmpo_expval.jl")

#from build_tmpo_[fw|expval].jl
export
    build_fw_tMPO_regul_beta, 
    build_ising_fw_tMPO_regul_beta, 
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
include("tmpo/ising_fold_tmpo.jl")

#from build_fold_tmpo.jl
export 
     #build_ising_folded_tMPO,  # superseded by build_ham_ ? 
     build_ising_folded_tMPS,
     build_folded_tMPO_regul_beta, 
     build_folded_left_tMPS,
     build_ham_folded_tMPO,
     build_folded_open_tMPO,
     build_folded_tMPO_new


include("power_method/pm.jl")
include("power_method/symm_pm.jl")

#from power_method.jl
export powermethod, 
    powermethod_converge_norm,
    powermethod_converge_eig,
    powermethod_fold,
    powermethod_Lonly,
    pm_all,
    pm_svd


export powermethod_sym,
    powermethod_sym_norms,
    powermethod_sym_rdm


include("lightcone/expvals_lr.jl")
export expval_LR

include("lightcone/cone.jl")

export evolve_cone,
 init_cone_ising


export init_cone, run_cone

# GPU but I really don't know 
export gpu_run_cone, 
gpu_run_cone_svd,
gpu_truncate_sweep,
gpu_expval_LR,
cpu_expval_LR,
gpu_expval_LL_sym

end #module ITransverse