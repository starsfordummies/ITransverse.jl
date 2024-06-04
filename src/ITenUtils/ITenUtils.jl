module ITenUtils

using LinearAlgebra
using ITensors

using NDTensors 

import NDTensors:
 replace_nothing,
 default_use_absolute_cutoff,
 default_use_relative_cutoff,
 expose,
 truncate!!


include("pparams.jl")

include("utils.jl")
include("matrix_utils.jl")
include("moreutils.jl")


include("ctruncate.jl")
include("ceigen.jl")

# Symmetric SVD/EIG decompositions
include("gen_svdeig_symm.jl")


#include("symmsvd_iten.jl")

include("bench_data.jl")




# from pparams
export pparams, ppm_params, trunc_params, model_params, tmpo_params

#from utils.jl
export myrMPS, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor,
    normbyfactor
    # check_diag_matrix, 
    # check_id_matrix,
    # isid


# moreutils.jl
export randsymITensor,
    isid,
    #plot_matrix,
    pinvten,
    #symmetrize,
    randITensor_decayspec

# matrix_utils.jl
export symmetrize,
    check_id_matrix,
    check_diag_matrix,
    randmat_decayspec

# gen_svdeig_symm.jl
export
    symm_svd,
    symm_oeig,
    mytrunc_svd,
    mytrunc_eig

end
