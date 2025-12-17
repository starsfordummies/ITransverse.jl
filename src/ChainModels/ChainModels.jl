module ChainModels

using ITensors, ITensorMPS

using ..ITenUtils

include("model_params.jl")
include("generic.jl")
include("id_mpo.jl")
include("ising_parallel.jl")
include("potts.jl")
include("xxzmodel.jl")
include("random_mpo.jl")
include("exph_generic.jl")

export ModelParams, IsingParams, PottsParams, XXZParams, NoParams

# export  expH, expHim

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

# from xxzmodel.jl
#export expH_XXZ_2o

export build_H, build_Ut

export timeEvo_MPO_2ndOrder, timeEvo_MPO_2ndOrder_LRflipped

end
