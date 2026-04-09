using ITensors
using ITensorMPS

using ITransverse

tp = ising_tp(init_state=[1,0,0,1]/2)


b = FoldtMPOBlocks(tp)

itermax = 500
eps_converged=1e-9

cutoff = 1e-12
maxdim = 128
direction = :right
alg = "densitymatrix"

truncp = (;cutoff, maxdim, direction, alg)


evs = [] 

rvecs = []
ds2s = []
r2s = [] 


ts = 60 + tp.nbeta
alltimes = ts.* tp.dt

Nsteps = ts

time_sites = siteinds(4, Nsteps)

init_mps = folded_right_tMPS(b, time_sites)

mpo_1 = folded_tMPO(b, time_sites)

pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="norm")
rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 
norm(rr)
overlap_noconj(rr,rr)


pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="overlap")
rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 
norm(rr)
overlap_noconj(rr,rr)


pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="nothing")
rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 
norm(rr)
overlap_noconj(rr,rr)

overlap_noconj(rr,rr)/norm(rr)


pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="overlap")
rr, ds2_pm  = powermethod_sym(init_mps, mpo_1, pm_params) 
norm(rr)
overlap_noconj(rr,rr)


pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="nothing")
rrn, ds2_pm  = powermethod_sym(rr, mpo_1, pm_params) 
norm(rrn)
overlap_noconj(rrn,rrn)


pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="nothing")
rrn, ds2_pm  = powermethod_sym(rrn, mpo_1, pm_params) 
norm(rrn)
overlap_noconj(rrn,rrn)

ortho_lims(rr)
ortho_lims(rrn)