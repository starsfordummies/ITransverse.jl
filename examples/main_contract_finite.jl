using ITensors, ITensorMPS
using ITransverse
using ProgressMeter
#using ITransverse.ITenUtils

using ITransverse: vI

""" Contract a finite (folded) system in the transverse direction """
function contract_strings(N::Int, length_string::Int, ts::Int, nbeta::Int)

    mp = IsingParams(1, 2., 0.)
    tp = tMPOParams(0.1, build_expH_ising_murg, mp, nbeta, [1,0])

    cutoff = 1e-12
    maxbondim = 128
  
    truncp = TruncParams(cutoff, maxbondim)

    P_up = ComplexF64[1,0,0,0]

    b = FoldtMPOBlocks(tp)
    
    time_sites = siteinds(4, ts + 2*nbeta)

    tmpo_id  = folded_tMPO(b, time_sites; fold_op = vI) 
    tmpo_Pz  = folded_tMPO(b, time_sites; fold_op = P_up) 

    left_tmps = folded_left_tMPS(b, time_sites; fold_op = vI)

    right_tmps = folded_right_tMPS(b, time_sites; fold_op = vI)
    #=  If we want thermo limit, find dominant first ? 
    pm_params = PMParams(truncp, 400, eps_converged, true, "RDM")

    init_mps, _ = powermethod_sym(init_mps, tmpo_id, pm_params)
  
    @info "init overlap =", ITransverse.overlap_noconj(init_mps, init_mps)
    @info "norm MPO_1 =", norm(tmpo_id), 0.5* norm(tmpo_id)
    =#


    op_string = fill("Id", N)

    mpo_list = fill(tmpo_id, N)

    for ii in 1:N
        if op == "Pz"
            mpo_list[ii] = tmpo_Pz
        end
    end 

    trunc_method = "RDM"

    overlap = contract_tn_transverse(left_tmps, mpo_list, right_tmps)
    # @info normalization
    # @info temp1
    # @info temp2

    # lognormalization = log(temp1) + sum(log.(temp2))

    #     push!(trt2s, rtm2_contracted(psiOR, psiOR, normalize_factor=overlap_noconj(psiOR,psiOR)))

    #     next!(p; showvalues = [(:Info, "$(ev) $(op_string)" )])

    # end

    return overlap
end


eevs = []
for ts = 40:4:40
    ov = contract_strings(12, 1, ts, 0)
    push!(eevs, ov)
end

