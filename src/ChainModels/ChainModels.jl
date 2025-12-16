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

# export  build_expH, build_expHim

export up_state, down_state, plus_state
export vX, vZ, vI

# from ising.jl
export build_H_ising, 
    build_expH_ising_murg,
    build_expH_ising_symm_svd,
    build_expH_ising_murg_4o

#from potts.jl
export build_H_potts_manual,
    build_H_potts,
    #build_expH_potts_2o,
    build_expH_potts_murg,
    build_expH_potts_symmetric_svd

export build_expH_random_symm_svd_1o

# from xxzmodel.jl
export build_expH_XXZ_2o

export timeEvo_MPO_2ndOrder, timeEvo_MPO_2ndOrder_LRflipped

end
