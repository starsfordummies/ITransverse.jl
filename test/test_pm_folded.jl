using ITensors
using ITensorMPS

using ITransverse

using Test 

function main_folded_pm(tp, Nt, pm_params)

    b = FoldtMPOBlocks(tp)


    ts = Nt + tp.nbeta
    alltimes = ts.* tp.dt

    infos = Dict(:tp => tp, :pm_params => pm_params, :b => b, :times => alltimes)

    Nsteps = ts

    time_sites = siteinds(4, Nsteps)

    init_mps = folded_right_tMPS(b, time_sites)

    mpo_1 = folded_tMPO(b, time_sites)
    mpo_Z = folded_tMPO(b, time_sites; fold_op = vZ)

    ll, rr, infos  = powermethod_op(init_mps; mpo_id=mpo_1, mpo_op=mpo_Z, pm_params) 

    ev = compute_expvals(ll, rr, ["X","Z"], b)
    @show ev
  

    return ev, max(maxlinkdim(rr),maxlinkdim(ll))
end


@testset "Truncation sweeps for power method " begin 
Nt = 20
tp = ising_tp()

itermax = 400
stuck_after = 100
eps_converged=1e-6

cutoff = 1e-12
maxdim = 128

ref = ITransverse.tebd_z(Nt, tp; N=2*Nt, cutoff=1e-12)[end]
allerrs = [] 
allchis = []
allpars = []
for direction = [:right, :left]
    for alg = ["densitymatrix", "naive", "RTM", "naiveRTM"]
        for opt_method = [:nosym, :sym] 
            for normalization = ["norm", "overlap"]

            truncp = (;cutoff, maxdim, direction, alg)
            pm_params = PMParams(;truncp, itermax, eps_converged, opt_method, normalization, stuck_after)

            try
                ev, chi = main_folded_pm(tp, Nt, pm_params)

                push!(allerrs, real(ev["Z"]) - ref)
                push!(allchis, chi)
                push!(allpars, [direction, alg, opt_method, normalization])
            catch e 
                println("error $e")
            end

            end
        end
    end
end


# for (params, zerr, chi) in zip(allpars, allerrs, allchis)
#     @testset "$(params)" begin
#         @info chi
#         @test abs(zerr) < 1e-4
#     end
# end

# allerrs = [] 
# allchis = []
# allpars = []
# for direction = [:right, :left]
#     for alg = ["naiveRTM"]
#         for opt_method = [:nosym,:sym]
#             for normalization = ["norm", "overlap"]

#             truncp = (;cutoff, maxdim, direction, alg)
#             pm_params = PMParams(;truncp, itermax, eps_converged, opt_method, normalization, stuck_after)

#             try
#                 ev, chi = main_folded_pm(tp, Nt, pm_params)

#                 push!(allerrs, real(ev["Z"]) - ref)
#                 push!(allchis, chi)
#                 push!(allpars, [direction, alg, opt_method, normalization])
#             catch e 
#                 println("error $e")
#             end

#             end
#         end
#     end
# end


for (params, zerr, chi) in zip(allpars, allerrs, allchis)
    @info params, " ", chi, "  ", abs(zerr)
    if params[1] == :left
        @test abs(zerr) < 1e-5
    else
        @test abs(zerr) < 2e-2
    end
end


end