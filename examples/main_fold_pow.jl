using ITensors
using ITensorMPS

using ITransverse
using ITransverse: vX, vZ, vI

function main_folded_pm()

    tp = ising_tp()

    b = FoldtMPOBlocks(tp)

    itermax = 500
    eps_converged=1e-9

    cutoff = 1e-12
    maxdim = 128
    direction = :right
    alg = "densitymatrix"

    truncp = (;cutoff, maxdim, direction, alg)

    pm_params = PMParams(;truncp, itermax, eps_converged, opt_method=:sym, normalization="norm")

    evs = [] 

    rvecs = []
    ds2s = []
    r2s = [] 


    ts = 60 + tp.nbeta
    alltimes = ts.* tp.dt

    infos = Dict(:tp => tp, :pm_params => pm_params, :b => b, :times => alltimes)

    Nsteps = ts

    time_sites = siteinds(4, Nsteps)

    init_mps = folded_right_tMPS(b, time_sites)

    mpo_1 = folded_tMPO(b, time_sites)
    mpo_Z = folded_tMPO(b, time_sites; fold_op = vZ)


    ll, rr, ds2_pm  = powermethod_op(init_mps; mpo_id=mpo_1, mpo_op=mpo_Z, pm_params) 

    ev = compute_expvals(ll, rr, ["X","Z"], b)
    @show ev

    push!(rvecs, rr)
    push!(evs, ev)
    push!(ds2s, ds2_pm)

    push!(r2s, ITransverse.gen_renyi2(ll, rr))


    return rvecs, evs, ds2s, r2s, ts, infos
end



rvecs, evs, ds2s, r2s, alltimes, infos = main_folded_pm()

println(evs)
