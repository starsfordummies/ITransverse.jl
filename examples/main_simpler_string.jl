!true && include("../src/ITransverse.jl")
!true && include("../src/ChainModels/ChainModels.jl")

using ITensors, JLD2
using ITransverse
using ProgressMeter
#using ITransverse.ITenUtils

function simpler_string(length_string::Int,ts::Int, nbeta::Int; Jxx::Real=1,hz::Real, hx::Real=0.0)

    tp = tMPOParams(0.1, build_expH_ising_murg, IsingParams(Jxx, hz, hx), nbeta, [1,0])
    tp_proj = tMPOParams(tp; tr=[1,0,0,0])

    cutoff = 1e-12
    maxbondim = 128
    #length_string = 10
    #itermax = 10
    eps_converged=1e-6

    truncp = TruncParams(cutoff, maxbondim)

    Id   = ComplexF64[1,0,0,1]
    P_up = ComplexF64[1,0,0,0]

    ev = [] 

    tpim = tMPOParams(tp; dt=-im*tp.dt)

    b = FoldtMPOBlocks(tp)
    b_im = FoldtMPOBlocks(tpim)
    
    time_sites = siteinds(4, ts + 2*nbeta)

    tmpo_id  = ITransverse.folded_tMPO_doublebeta(b, b_im, time_sites) # , P_up)

    init_mps = folded_right_tMPS(tmpo_id)
    init_mps = ITransverse.normbyfactor(init_mps, sqrt(ITransverse.overlap_noconj(init_mps, init_mps))) 


    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, 400, eps_converged, true, "RDM")

    init_mps, _ = powermethod_sym(init_mps, tmpo_id, pm_params)

    @info "init overlap =", ITransverse.overlap_noconj(init_mps, init_mps)
    @info "norm MPO_1 =", norm(tmpo_id), 0.5* norm(tmpo_id)



    # Build an operator string long `length_string`, populate it with increasingly more Pz
    # and feed it to expval_LR_apply_list_sym.  The first in the list get applied first, so we should start from Id's! 

    op_string = fill("Id", length_string)

    trunc_method = "RDM"

    #normalization, _ = ITransverse.expval_LR_apply_list_sym(init_mps, op_string, "Id", b, b_im; method="RTM", maxdim=64)
    normalization, temp1,temp2 = ITransverse.expval_LR_apply_list_sym_2(init_mps, op_string, b, b_im; method=trunc_method, maxdim=maxbondim)

    @info normalization
    @info temp1
    @info temp2

    lognormalization = log(temp1) + sum(log.(temp2))

    @info "should be ~1 ? ", normalization , lognormalization

    allevs = []
    logevs = []
    len_pz = []
    trt2s = []

    p = Progress(length_string; desc="L=$(length(init_mps)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    for ii = 1:length_string
        op_string[end-ii+1] = "Pz"
        how_many_pz = ii*2

        #@info ii, op_string

        #ev, _ = ITransverse.expval_LR_apply_list_sym(init_mps, op_string, "Pz", b, b_im, method="RTM")
        ev, temp1, temp2, psiOR  = ITransverse.expval_LR_apply_list_sym_2(init_mps, op_string, b, b_im, method=trunc_method, maxdim=maxbondim)
        logev = log(temp1) + sum(log.(temp2))
        @info ev
        @info temp1
        @info temp2

        push!(allevs, ev/normalization)
        push!(logevs, logev - lognormalization)
        push!(len_pz, how_many_pz)

        push!(trt2s, rtm2_contracted(psiOR, psiOR, normalize_factor=overlap_noconj(psiOR,psiOR)))

        next!(p; showvalues = [(:Info, "$(ev) $(op_string)" )])

    end

    return len_pz, allevs, logevs, trt2s
end


eevs = []
for ts = 60:4:60
    lens_pz, allevs, logevs, trt2s = simpler_string(12, ts, 4, hz=1.0)
    push!(eevs, allevs, logevs, trt2s)
end

