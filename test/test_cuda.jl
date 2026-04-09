using ITensors, ITensorMPS
using ITransverse
using Test
using ITransverse: togpu

try
    using CUDA: CUDA as CUDA
    CUDA.functional() || (println("No CUDA-capable GPU found, skipping"); return)
catch
    println("CUDA.jl not available, skipping"); return
end


# Skip entire file gracefully if CUDA is not available / no GPU present
CUDAext = Base.get_extension(ITransverse, :ITransverseCUDAExt)
if isnothing(CUDAext) || !(@isdefined(togpu))
    @warn "ITransverseCUDAExt not loaded — skipping CUDA tests"

else

@testset "CUDA: adapt tMPOParams / block structs" begin

    tp = ising_tp()

    @testset "tMPOParams → GPU" begin
        tp_gpu = togpu(tp)
        @test NDTensors.unwrap_array_type(tp_gpu.bl) <: CUDA.CuArray
        tp_cpu = tocpu(tp_gpu)
        @test NDTensors.unwrap_array_type(tp_cpu.bl) <: Array
    end

    @testset "FwtMPOBlocks → GPU" begin
        b = FwtMPOBlocks(tp)
        b_gpu = togpu(b)
        @test NDTensors.unwrap_array_type(b_gpu.Wc) <: CUDA.CuArray
        b_cpu = tocpu(b_gpu)
        @test NDTensors.unwrap_array_type(b_cpu.Wc) <: Array
    end

    @testset "FoldtMPOBlocks → GPU" begin
        b = FoldtMPOBlocks(tp)
        b_gpu = togpu(b)
        @test NDTensors.unwrap_array_type(b_gpu.WWc) <: CUDA.CuArray
        b_cpu = tocpu(b_gpu)
        @test NDTensors.unwrap_array_type(b_cpu.WWc) <: Array
    end
end

@testset "CUDA: power method (small)" begin

    tp = ising_tp()
    b  = FoldtMPOBlocks(togpu(tp))

    Nsteps = 20
    time_sites = siteinds(4, Nsteps)

    init_mps   = folded_right_tMPS(b, time_sites)
    mpo_1      = folded_tMPO(b, time_sites)
    mpo_op      = folded_tMPO(b, time_sites; fold_op=vZ)


    truncp    = (; cutoff=1e-10, maxdim=32, direction=:right, alg="RTM")
    pm_params = PMParams(; truncp, itermax=50, eps_converged=1e-6, opt_method=:sym, normalization="norm")

    ll, rr, _ = powermethod_op(init_mps; mpo_id=mpo_1, mpo_op=mpo_1, pm_params)

    @test NDTensors.unwrap_array_type(ll[1]) <: CUDA.CuArray
    ev = compute_expvals(ll, rr, ["Z"], b)
    @test abs(imag(ev["Z"])) < 1e-6
end

@testset "CUDA: run_cone (small)" begin

    tp = ising_tp()
    b  = FoldtMPOBlocks(togpu(tp))
    c0 = init_cone(b, 4)

    truncp     = (; cutoff=1e-8, maxdim=32, direction=:right, alg="RTM")
    cone_pars  = ConeParams(; truncp, opt_method=:sym, optimize_op=vZ)

    cp = DoCheckpoint("_test_cuda_cone.jld2"; params=Dict(), save_at=Int[],
        f_obs      = (overlap = s -> overlap_noconj(tocpu(s.L), tocpu(s.R)),),
        f_savestate= (L = s -> tocpu(s.L), R = s -> tocpu(s.R), b = s -> tocpu(s.b)))

    ll, rr, cp = run_cone(c0, b, cone_pars, cp, 14)

    @test length(ll) == 14
    @test NDTensors.unwrap_array_type(ll[1]) <: CUDA.CuArray

end

end
