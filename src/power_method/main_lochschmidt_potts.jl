if Base.Sys.islinux()
    println("On linux - setting MKL and 8 threads")
    using MKL
    using LinearAlgebra

    BLAS.set_num_threads(8)  # 8 threads seems to be a sweet spot

else
    using LinearAlgebra
    using Infiltrator

end
using Plots
using ITensors, FileIO, JLD2, Dates

# include("utils.jl")
# include("../models/potts.jl")
# include("../tmpo/build_tmpo.jl")
# include("compute_entropies.jl")
# include("../truncations/sweeps_trunc.jl")
# include("power_method.jl")

using ITransverse

if Base.Sys.isapple()
    ITensors.enable_debug_checks()
end

# Entropy should go like (?)
##  f1p(x) = 0.8/6 * log(sin(π*x))

function main() 


JXX = 1.0  
ff = 1.0  

dt = 0.1

nbeta = 10

SVD_cutoff = 1e-14    
maxbondim = 50
itermax = 100


Ntime_steps = 40
Nsteps = Ntime_steps +2*nbeta

# conserve_qns = false
time_sites =  addtags(siteinds("S=3", Nsteps), "time")
test_mps = productMPS(time_sites,"+")

mpo_L = build_potts_tMPO_regul_beta(build_expH_potts_2o, JXX, ff, dt, nbeta, time_sites)
mpo_R = swapprime(mpo_L, 0, 1, "Site")


#params = Dict("JXX" => JXX , "hz" => hz, "dt" => dt)
pm_params = Dict(:itermax => itermax, :SVD_cutoff=> SVD_cutoff, :maxbondim => maxbondim )



#ll_2o, rr_2o, ds2s_2o  = powermethod(test_mps, mpo_L, mpo_R, pm_params)
#ll_2o_test, rr_2o_test, ds2s_2o_t  = powermethod_fold(test_mps, mpo_L, mpo_L, pm_params)


time_sites =  addtags(siteinds("S=1", Nsteps), "time")
test_mps = productMPS(time_sites,"+")


println("murg")
mpo_L = build_potts_tMPO_regul_beta(build_expH_potts_murg, JXX, ff, dt, nbeta, time_sites)

println("murg alt")
mpo_L = build_potts_tMPO_regul_beta(build_expH_potts_murg_alt, JXX, ff, dt, nbeta, time_sites)

sleep(100)

mpo_R = swapprime(mpo_L, 0, 1, "Site")


# Cannot work if the MPO is not symmetric L<-->R ! 
#ll_s, ds2s  = powermethod_sym(test_mps, mpo_L, pm_params)

# ???
#ll_s, ds2s  = powermethod_svd(test_mps, mpo_L, pm_params)

ll, rr, ds2s  = powermethod(test_mps, mpo_L, mpo_R, pm_params)
ll_test, rr_test, ds2s_t  = powermethod_fold(test_mps, mpo_L, mpo_L, pm_params)

#re_plot = plot(real(generalized_entropy(ll_2o,rr_2o))[nbeta+1:Ntime_steps+nbeta-1], label="re,2o");
#re_plot = plot!(real(generalized_entropy(ll_2o_test,rr_2o_test))[nbeta+1:Ntime_steps+nbeta-1], label="2o,fold"); 

re_plot = plot!(real(generalized_entropy(ll,rr))[nbeta+1:Ntime_steps+nbeta-1], label="murg");
re_plot = plot!(real(generalized_entropy(ll_test,rr_test))[nbeta+1:Ntime_steps+nbeta-1], label="murg,fold");

#plot!(real(generalized_entropy(ll_s,conj(ll_s))), label="re alt svd") |> display


#im_plot = plot(imag(generalized_entropy(ll_2o,rr_2o))[nbeta+1:Ntime_steps+nbeta-1], label="im,2o");  ## |> display
#im_plot = plot!(imag(generalized_entropy(ll_2o_test,rr_2o_test))[nbeta+1:Ntime_steps+nbeta-1], label="2o,fold"); 

im_plot = plot!(imag(generalized_entropy(ll,rr))[nbeta+1:Ntime_steps+nbeta-1],label ="murg"); 

im_plot = plot!(imag(generalized_entropy(ll_test,rr_test))[nbeta+1:Ntime_steps+nbeta-1],label="murg,fold"); 

#plot!(imag(generalized_entropy(ll_s,conj(ll_s))), label="im alt svd") |> display


#ds2_plot = plot(log.(ds2s_2o),label="2o");
#ds2_plot = plot!(log.(ds2s_2o_t),label="2o,fold"); 
ds2_plot = plot!(log.(ds2s_t),label="murg");
ds2_plot = plot!(log.(ds2s_t),label="murg,fold"); 


if Base.Sys.isapple()
    display(re_plot)
    display(im_plot)
    display(ds2_plot)
else
    png(re_plot,"out_re.png")
    png(im_plot,"out_im.png")
    png(ds2_plot,"out_ds2.png")

    println(generalized_entropy(ll_2o,rr_2o))
    println("###")
    println(generalized_entropy(ll,rr))

end


end


ITensors.space(::SiteType"S=3") = 7
ITensors.state(::StateName"↑", ::SiteType"S=3") = [1, 0, 0, 0, 0, 0, 0]
ITensors.state(::StateName"+", ::SiteType"S=3") = [1,1,1,1,1,1,1]/sqrt(7)

ITensors.state(::StateName"+", ::SiteType"S=1") = [1,1,1]/sqrt(3)


main()