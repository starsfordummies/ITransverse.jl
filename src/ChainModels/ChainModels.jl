module ChainModels

using ITensors, ITensorMPS

using ..ITenUtils

include("generic.jl")
include("ising.jl")
include("ising_parallel.jl")
include("potts.jl")
include("xxzmodel.jl")

export  build_expH, build_expHim

# from ising.jl
export build_H_ising_manual,
    build_H_ising, 
    build_expH_ising_1o, 
    build_expH_ising_2o, 
    build_expH_ising_murg
    

# from ising_parallel.jl
export  build_expH_ising_parallel_field,
 build_expH_ising_parallel_field_murg

#from potts.jl
export build_H_potts_manual,
    #build_H_potts_manual_lowtri,
    build_H_potts,
    build_expH_potts_2o,
    build_expH_potts_murg,
    build_expH_potts_symmetric_svd


# from xxzmodel.jl
export build_H_XXZ_manual,
    #build_H_XXZ_manual_lowtri,
    build_H_XXZ,
    #build_expH_XXZ_1o,
    #build_expH_XXZ_2o,
    build_expH_XXZ_murg,
    build_expH_XXZZ_murg_from_ising

export ising_tp


end
