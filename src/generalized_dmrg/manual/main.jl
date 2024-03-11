include("./1_util.jl")
include("./2_trunc.jl")
include("./3_canform.jl")
include("./4_envs.jl")
include("./5_factorize.jl")
include("./6_dmrg.jl")


function main_gen(nsweeps=3, fact_method=2)

    aa = siteinds("S=1/2",16)
    
    #hmpo = build_H_ising(aa, 1., 1.)
 
    hmpo = build_expH_ising_murg(aa, 1., 1., 0.1)

    psi, en = mygendmrg_manual(hmpo, aa; nsweeps, fact_method)

    return psi, en 
end

function main_dmrg()
    println("Standard DMRG")

    aa = siteinds("S=1/2",16)
    
    #hmpo = build_H_ising(aa, 1., 1.)
 
    hmpo = build_expH_ising_murg(aa, 1., 1., 0.1)

    startmps = randomMPS(aa)

    en, psi = dmrg(hmpo, startmps, nsweeps=3, which_decomp="svd",  #eigen gives the same 
    eigsolve_which_eigenvalue=:LM, ishermitian=false, cutoff=1e-14, maxdim=100, 
    eigsolve_maxiter=4, eigsolve_krylovdim=5, outputlevel=1)

    return psi, en

end

psi1, en1 = main_gen(4, 1)
psi2, en2 = main_gen(4, 2)
psi3, en3 = main_dmrg()

@show en1[end], maxlinkdim(psi1)
@show en2[end], maxlinkdim(psi2)
@show en3, maxlinkdim(psi3)
