using ITensors
using ITensorMPS
using ITransverse
using ITransverse: plus_state, up_state, vX, vZ

using Test

""" Builds the dominant vector for unfolded Ising using power method"""
function ising_fwb(tp::tMPOParams, TT::Int)

    cutoff = 1e-12
    maxdim = 256
    itermax = 200
    eps_converged = 1e-9

    truncp = TruncParams(cutoff, maxdim)
    pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "overlap")

    @info ("Optimizing for T=$(TT) with $(tp.nbeta) imag steps ")
    @info ("Initial state $(tp.bl)")


    b = FwtMPOBlocks(tp)

    Ntime_steps = TT
    Nsteps = Ntime_steps 

    time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")

    mpo = fwback_tMPO(b, time_sites, tr=tp.bl)
    start_mps = fwback_tMPS(b, time_sites; tr=tp.bl, LR=:right)

    psi_trunc, ds2 = powermethod_sym(start_mps, mpo, pm_params)


    # # Entropies 
    sgen = generalized_vn_entropy_symmetric(psi_trunc)
    # sgen_sv = generalized_svd_vn_entropy_symmetric(psi_trunc)

    # tsallis_gen = ITransverse.generalized_r2_entropy_symmetric(psi_trunc)

    #svn = vn_entanglement_entropy(psi_trunc)

    leading_eig = inner(conj(psi_trunc'), mpo, psi_trunc)

    # silly extra check so we can see that (LTTR) = lambda^2 (LR)
    OL = apply(mpo, psi_trunc,  alg="naive", truncate=false)
    leading_sq = overlap_noconj(OL, OL)

    normalization = overlap_noconj(psi_trunc,psi_trunc)
    leading_eig, leading_sq, normalization

    return psi_trunc, b


end





function ffolded(tp::tMPOParams, TT::Int)
    b = FoldtMPOBlocks(tp)

    cutoff = 1e-12
    maxdim = 256
    itermax = 500
    eps_converged=1e-9

    truncp = TruncParams(cutoff, maxdim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RDM", "overlap")


    Nsteps = TT

    time_sites = siteinds(4, Nsteps)

    init_mps = folded_right_tMPS(b, time_sites)

    mpo_X = folded_tMPO(b, time_sites; fold_op=vX)
    mpo_Z = folded_tMPO(b, time_sites; fold_op=vZ)

    mpo_1 = folded_tMPO(b, time_sites)

    ll, rr, ds2_pm  = powermethod_op(init_mps, mpo_1, mpo_X, pm_params) 

    return ll, rr, ds2_pm, b
end




@testset "folded/unfolded" begin
    

JXX = 1.0
hz = 0.7
gx = 0.0
#H= JXX - 2.0 * 0.525 Z + 2 * 0.25 X


dt = 0.1

# init_state = plus_state
init_state = up_state

mp = IsingParams(JXX, hz, gx)

tp = tMPOParams(dt,  expH_ising_murg, mp, 0, init_state)

diffs = []

for TT = 10:10:30

    psi, b = ising_fwb(tp,2*TT)
    tmpo_z = fwback_tMPO(b, siteinds(psi); mid_op = [1,0,0,-1], tr=b.tp.bl)
    ev_unfold = expval_LR(psi, tmpo_z, psi)/overlap_noconj(psi,psi)


    ##### Folded 

    ll,rr, ds2_pm, bf = ffolded(tp, TT)

    ll = replace_siteinds(ll, siteinds(rr))
    mpo_Z = folded_tMPO(bf, siteinds(rr); fold_op=vZ)


    ev_fold = expval_LR(ll, mpo_Z, rr)/overlap_noconj(ll,rr)

    @show TT, ev_fold, ev_unfold 

    @test abs.(ev_unfold - ev_fold) < 0.001

end

end