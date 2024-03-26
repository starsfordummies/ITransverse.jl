module MyModels
using ITensors

include("../myutils/myutils.jl")
#using .MyUtils

include("brakets.jl")
include("ising.jl")
include("potts.jl")
include("xxzmodel.jl")

# from ising.jl
export build_H_ising_manual,
    build_H_ising_manual_lowtri, 
    build_H_ising, 
    build_H_ising_ZZ_X,
    build_H_ising_YY,
    build_expH_ising_1o, 
    build_expH_ising_2o, 
    build_expH_ising_murg, 
    build_expH_ising_murg_ZZX, 
    build_expH_ising_murg_YY

#from potts.jl
export build_H_potts_manual,
    build_H_potts_manual_lowtri,
    build_H_potts,
    build_expH_potts_2o,
    build_expH_potts_murg

# from xxzmodel.jl
export build_H_XXZ_manual,
    build_H_XXZ_manual_lowtri,
    build_H_XXZ,
    build_expH_XXZ_1o,
    build_expH_XXZ_2o,
    build_expH_XXZ_murg

end
