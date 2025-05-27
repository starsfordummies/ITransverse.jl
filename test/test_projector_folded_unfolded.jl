using ITensors, ITensorMPS
using ITransverse
using Test

using ITransverse: up_state


@testset "Testing that folded+projector is the same as amplitude^2 using transverse contraction " begin

tp = ising_tp()
maxbondim=100

Ntime_steps = 30

nbeta = 0
Nsteps = nbeta + Ntime_steps + nbeta

time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")
time_sites_fold = addtags(siteinds(4, Nsteps; conserve_qns=false), "time")


#init_state = up_state #  rand(ComplexF64, 2)
init_state = rand(ComplexF64, 2)

mp = IsingParams(1, 0.7, 0)

tp = tMPOParams(tp; nbeta, mp=mp, bl=init_state)

Nsteps = nbeta + Ntime_steps + nbeta

b= FwtMPOBlocks(tp)


b_fold = FoldtMPOBlocks(tp)

mpo = fw_tMPO(b, time_sites; tr = ITensor(up_state, Index(2)))

left_mps = ITransverse.fw_left_tMPS(b, time_sites; tr = up_state)
right_mps = ITransverse.fw_right_tMPS(b, time_sites; tr = up_state)

mpo_fold = folded_tMPO(b_fold, time_sites_fold, fold_op = [1,0,0,0])

left_fold = folded_left_tMPS(b_fold, time_sites_fold, fold_op = [1,0,0,0])
right_fold = folded_right_tMPS(b_fold, time_sites_fold, fold_op = [1,0,0,0])

 abs2.(overlap_noconj(left_mps, right_mps))
  overlap_noconj(left_fold, right_fold)#< 1e-5

@test abs( abs2.(overlap_noconj(left_mps, right_mps)) - overlap_noconj(left_fold, right_fold)) < 1e-5
# Contract unfolded network 
ll = left_mps
rr = right_mps
for nn = 1:4
    ll = applys(mpo, ll; cutoff=1e-12, maxdim=maxbondim)
    rr = apply(mpo, rr; cutoff=1e-12, maxdim=maxbondim)
end
maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)

Lsq = abs2(overlap_noconj(ll,rr))

# Contract folded network 
ll = left_fold
rr = right_fold
for nn = 1:4
    ll = applys(mpo_fold, ll; cutoff=1e-12, maxdim=maxbondim)
    rr = apply(mpo_fold, rr; cutoff=1e-12, maxdim=maxbondim)
end
maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)

@test abs.(Lsq - overlap_noconj(ll,rr))/Lsq < 1e-4

end





@testset "Overlap network with/without folding for XXZ Heisenberg spin 1 " begin

dt=0.5
Ntime_steps = 5
nbeta = 0
Nsteps = nbeta + Ntime_steps + nbeta


mp_xxz = XXZParams(1, 0.7, 0.0)

tp_xxz = tMPOParams(dt, ITransverse.ChainModels.build_expH_XXZ_2o_spin1, mp_xxz, nbeta, [1,0,0])

rotated_phys = 7

time_sites = addtags(siteinds(rotated_phys, Nsteps; conserve_qns=false), "time")
time_sites_fold = addtags(siteinds(rotated_phys^2, Nsteps; conserve_qns=false), "time")

init_state = [1,0,0] # rand(ComplexF64, 3)

b= FwtMPOBlocks(tp_xxz)

mpo = fw_tMPO(b, time_sites; tr = [1,0,0])

left_mps = ITransverse.fw_left_tMPS(b, time_sites; tr = [1,0,0])
right_mps = ITransverse.fw_right_tMPS(b, time_sites; tr = [1,0,0])

# Contract unfolded network 
ll = left_mps
rr = right_mps
for nn = 1:1
    ll = applyns(mpo, ll) # ;cutoff=1e-12, maxdim=maxbondim)
    rr = applyn(mpo, rr) #; cutoff=1e-12, maxdim=maxbondim)
end
maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)

Lsq = abs2(overlap_noconj(ll,rr))


# Now folded 

b_fold = FoldtMPOBlocks(tp_xxz; init_state= kron(init_state, conj(init_state)));

mpo_fold = folded_tMPO(b_fold, time_sites_fold) # , fold_op = [1,0,0,0,0,0,0,0,0])

left_fold = folded_left_tMPS(b_fold, time_sites_fold)
right_fold = folded_right_tMPS(b_fold, time_sites_fold)

fidelity(left_fold,right_fold)
inner(left_fold,right_fold)
overlap_noconj(left_fold,right_fold)

 abs2.(overlap_noconj(left_mps, right_mps))
  overlap_noconj(left_fold, right_fold)#< 1e-5

@test abs( abs2.(overlap_noconj(left_mps, right_mps)) - overlap_noconj(left_fold, right_fold)) < 1e-5


# Contract folded network 
ll = left_fold
rr = right_fold
for nn = 1:1
    ll = applyns(mpo_fold, ll)# ; cutoff=1e-10, maxdim=maxbondim)
    rr = applyn(mpo_fold, rr) # ; cutoff=1e-10, maxdim=maxbondim)
end
maxlinkdim(ll)
maxlinkdim(rr)

overlap_noconj(ll,rr)

inner(ll,rr)

fidelity(ll,rr)

@test abs.(Lsq - overlap_noconj(ll,rr))/Lsq < 1e-4

end
