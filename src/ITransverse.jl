module ITransverse

using LinearAlgebra
using NDTensors
using ITensors, ITensorMPS
using ProgressMeter


using IGensors

include("ExtraUtils/ExtraUtils.jl")
using .ExtraUtils

# # from pparams.jl
export pparams, ppm_params, trunc_params

# #from utils.jl
export myrMPS, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor
    # check_diag_matrix, 
    # check_id_matrix,
    # isid
    vn_entanglement_entropy_cut, 
    vn_entanglement_entropy,
    renyi_entanglement_entropy,
    generalized_entropy,
    generalized_renyi_entropy

# #from compute_entropies.jl


include("truncations/sweeps.jl")
include("truncations/alt_sweeps.jl")

include("truncations/sweeps_sym.jl")

#from sweeps.jl
export truncate_normalize_sweep, truncate_normalize_sweep_LR
# from alt_sweeps.jl
export truncate_sweep_keep_lenv, truncate_sweep_aggressive_normalize
# sweeps_sym.jl
export truncate_normalize_sweep_sym!, truncate_normalize_sweep_sym_right


# from symm_decompositions.jl
#export symm_svd, symm_oeig


# include("models/ising.jl")
# include("models/potts.jl")
# include("models/xxzmodel.jl")

include("ChainModels/ChainModels.jl")
using .ChainModels

export pparams, ppm_params, trunc_params
export build_expH_ising_murg, build_expH_potts_murg, build_expH_potts_symmetric_svd

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



#from build_fold_tmpo.jl
export build_ising_folded_tMPO,
     build_ising_folded_tMPS,
     build_folded_tMPO_regul_beta, 
     build_folded_left_tMPS


include("power_method/pm.jl")
include("power_method/symm_pm.jl")

#from power_method.jl
export powermethod, 
    powermethod_converge_norm,
    powermethod_converge_eig,
    powermethod_fold,
    powermethod_Lonly,
    pm_all


export powermethod_sym,
    powermethod_sym_norms,
    powermethod_sym_rdm


include("lightcone/cone.jl")

export evolve_cone,
 init_cone_ising

end #module ITransverse