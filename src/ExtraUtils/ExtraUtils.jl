module ExtraUtils

using LinearAlgebra
using ITensors

include("pparams.jl")
include("utils.jl")

include("entropies.jl")

include("gen_form_checks.jl")
include("gen_entropies.jl")

include("symmsvd_iten.jl")

include("bench_data.jl")


# from pparams
export pparams, ppm_params, trunc_params

#from utils.jl
export myrMPS, 
    overlap_noconj, 
    check_symmetry_itensor_mpo, 
    check_symmetry_itensor,
    normbyfactor
    # check_diag_matrix, 
    # check_id_matrix,
    # isid

#from compute_entropies.jl
export vn_entanglement_entropy_cut, 
    vn_entanglement_entropy, 
    renyi_entanglement_entropy_cut, 
    renyi_entanglement_entropy,
    generalized_entropy_symmetric_cut, 
    generalized_entropy_symmetric, 
    #generalized_entropy_cut,
    generalized_entropy,
    generalized_renyi_entropy

# from gen_form_checks
export check_gencan_left_phipsi

end
