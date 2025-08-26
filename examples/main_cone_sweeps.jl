using ITensors
using ITensorMPS
using JLD2
using ITransverse.ITenUtils
using ProgressMeter
#using Plots

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state
using ITransverse: build_cols_cone, contract_cols, initialize_envs_rdm, overlap_envs, extend_cone!


function main_cone_sweeps(Nsteps::Int, n_ext::Int=2)

    JXX = 1.0  
    hz = -1.05 # 1.05
    gx = 0.5 # 0.5

    dt = 0.1

    nbeta = 0

    optimize_op = vZ
    init_state = up_state

    truncp_tiny = TruncParams(1e-12, 16)
    truncp_rtm = TruncParams(1e-12, 256, "right")

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state)

    # c0, b = init_cone(tp)
    b = FoldtMPOBlocks(tp)

    cc = build_cols_cone(b, Nt; fold_op=optimize_op, vwidth=1)


    left_envs, right_envs = initialize_envs_rdm(cc, truncp_tiny; verbose=false)
    ampli, stds = overlap_envs(left_envs, right_envs)

    @show maxlinkdim(left_envs)

    all_ts = [] 
    all_values = []
    all_chis = []

    for jj = 1:Nsteps

        # Staggered extend cone seems to give most reliable results, while not the fastest 
        if n_ext > 1
            extend_cone!(b, cc, left_envs, right_envs; fold_op=optimize_op, vwidth=n_ext-1)
        end
        extend_cone!(b, cc, left_envs, right_envs; fold_op=optimize_op, vwidth=1)

        # Make two rebuild env sweeps to be sure, maybe not necessary 
        sweep_rebuild_envs_rtm!(left_envs, right_envs, cc, truncp_rtm; verbose=false)
        sweep_rebuild_envs_rtm!(left_envs, right_envs, cc, truncp_rtm; verbose=false)

        ampli, stds = overlap_envs(left_envs, right_envs)

        @show maxlinkdim(left_envs)
        @show ampli

        push!(all_ts, length(left_envs[div(length(left_envs),2)]))
        push!(all_values, ampli)
        push!(all_chis, maxlinkdim(left_envs))

        @info all_ts .* dt
        @info all_values
        @info all_chis
    end

    return all_ts .* dt, all_values, all_chis
    end

times, expvals, chis = main_cone_sweeps(10, 2)
