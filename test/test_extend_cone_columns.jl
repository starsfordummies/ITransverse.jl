using ITensors
using ITensorMPS
using ITransverse.ITenUtils
using ProgressMeter
using Test

using ITransverse
using ITransverse: vX, vZ, vI, plus_state, up_state
using ITransverse: build_cols_cone, contract_cols, extend_cone!
#= 
@testset "Environments for light cone" begin
    Nt = 3

    JXX = 1.0  
    hz = 0.4 # 1.05
    gx = 0.0 # 0.5

    dt = 0.1

    nbeta = 0

    optimize_op = vZ
    init_state = up_state

    truncp_0 = TruncParams(1e-12, 512)

    truncp_tiny = TruncParams(1e-12, 16)
    truncp_rtm = TruncParams(1e-12, 256, "right")

    #time_sites = siteinds("S=3/2", 1)

    mp = IsingParams(JXX, hz, gx)
    #tp = tMPOParams(dt, build_expH_ising_murg, mp, nbeta, init_state, Id)
    tp = tMPOParams(dt, ITransverse.ChainModels.build_expH_ising_symm_svd, mp, nbeta, init_state)

    c0, b = init_cone(tp, Nt)

    tebd_ev = ITransverse.tebd_z(Nt, tp)

    c0_ev = expval_LR(c0, c0, [1,0,0,-1], b)


    cc = build_cols_cone(b, Nt; fold_op=[1,0,0,-1], vwidth=1)
    cols_ev = contract_cols(cc)
    left_envs, right_envs = initialize_envs_rdm(cc, truncp_tiny; verbose=false)
    envs_ev, stds_ev = overlap_envs(left_envs, right_envs)

    @test tebd_ev ≈ c0_ev rtol = 1e-6
    @test tebd_ev ≈ cols_ev rtol = 1e-6
    @test tebd_ev ≈ envs_ev rtol = 1e-6
    
    extend_cone!(b, cc, left_envs, right_envs; fold_op=[1,0,0,-1])
    extend_cone!(b, cc, left_envs, right_envs; fold_op=[1,0,0,-1])
    extend_cone!(b, cc, left_envs, right_envs; fold_op=[1,0,0,-1])

    cols_ext_ev = contract_cols(cc)
    tebd_ext_ev = ITransverse.tebd_z(Nt+3, tp)
    overlap_envs(left_envs, right_envs)
    
    sweep_rebuild_envs_rtm!(left_envs, right_envs, cc, truncp_rtm; verbose=false)
    envs_ext_ev, std_ext_ev = overlap_envs(left_envs, right_envs)

    @test tebd_ext_ev ≈ cols_ext_ev rtol = 1e-6
    @test tebd_ext_ev ≈ envs_ext_ev rtol = 1e-6
    @show tebd_ext_ev
    
    @test std_ext_ev < 1e-8
end
=# 