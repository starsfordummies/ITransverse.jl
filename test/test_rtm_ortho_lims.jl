using ITensors
using ITensorMPS
using ITensorMPS: isortho, orthocenter, ortho_lims
using ITransverse
using LinearAlgebra
using Random
using Test

# The RTM contractions build their outputs out of SVD isometries, so the results
# are canonical with the center on the site the sweep ends at (site 1 for
# direction=:left, the last site for :right). `setindex!` wipes the ortho limits
# while they are being written, so each routine restores them before returning.
#
# If those limits ever lied, `norm(psi)` would silently return
# `norm(psi[orthocenter])` instead of the true norm, corrupting every log-norm
# the environment machinery accumulates. Both properties are checked here.

# Deviation from isometry of the tensors the ortho limits *claim* are canonical.
function ortho_residual(psi::MPS)
    N = length(psi)
    res = 0.0
    for j in 1:min(ITensorMPS.leftlim(psi), N-1)     # expect T^dag T = I on the right link
        l = linkind(psi, j)
        isnothing(l) && continue
        T = psi[j] * dag(prime(psi[j], l))
        res = max(res, norm(Array(T, l', l) - I))
    end
    for j in max(ITensorMPS.rightlim(psi), 2):N      # expect T T^dag = I on the left link
        l = linkind(psi, j-1)
        isnothing(l) && continue
        T = psi[j] * dag(prime(psi[j], l))
        res = max(res, norm(Array(T, l', l) - I))
    end
    return res
end

function check_canonical(psi::MPS, expected_oc::Int)
    @test isortho(psi)
    @test orthocenter(psi) == expected_oc
    @test ortho_residual(psi) < 1e-10
    # the decisive one: the cheap norm(psi) path must agree with a full contraction
    @test norm(psi) ≈ sqrt(abs(real(inner(psi, psi)))) rtol = 1e-10
end

@testset "RTM contractions report correct ortho limits" begin
    Random.seed!(4321)
    n, chi = 8, 12

    @testset "T = $T, direction = $direction, maxdim = $maxdim" for
            T in (Float64, ComplexF64),
            direction in (:right, :left),
            maxdim in (8, 128)   # 8 truncates, 128 does not

        s  = siteinds(4, n)
        ψL = random_mps(T, s; linkdims=chi)
        ψR = random_mps(T, s; linkdims=chi)
        A  = T <: Complex ? random_mpo(s) + im * random_mpo(s) : random_mpo(s) + random_mpo(s)
        B  = T <: Complex ? random_mpo(s) + im * random_mpo(s) : random_mpo(s) + random_mpo(s)

        oc = direction == :right ? n : 1
        tp = (; alg="RTM", cutoff=1e-14, maxdim, direction)

        res = trapply(ψL, A, ψR; tp...)
        check_canonical(res.R, oc)

        res = tlapply(ψL, A, ψR; tp...)
        check_canonical(res.L, oc)

        res = tlrapply(ψL, A, B, ψR; tp...)
        check_canonical(res.L, oc)
        check_canonical(res.R, oc)
    end

    # naiveRTM gauges existing tensors instead of replacing them with isometries,
    # so its output is NOT canonical once truncation bites -- it must keep
    # reporting "no orthogonality center" rather than claim one.
    @testset "naiveRTM leaves the limits invalidated" begin
        s  = siteinds(4, n)
        ψL = random_mps(ComplexF64, s; linkdims=chi)
        ψR = random_mps(ComplexF64, s; linkdims=chi)
        A  = random_mpo(s) + im * random_mpo(s)

        res = tlapply(ψL, A, ψR; alg="naiveRTM", cutoff=1e-14, maxdim=8, direction=:right)
        @test !isortho(res.L)
        @test norm(res.L) ≈ sqrt(abs(real(inner(res.L, res.L)))) rtol = 1e-10
    end
end

# The other places that record ortho limits by hand, or touch MPS data below `setindex!`.
# Each claim is checked the same way: the tensors the limits call isometries must be
# isometries, and the cheap norm(psi) must agree with a full contraction.
norm_consistent(psi) = isapprox(norm(psi), sqrt(abs(real(inner(psi, psi)))); rtol = 1e-10)
valid_lims(psi) = 0 <= ITensorMPS.leftlim(psi) < ITensorMPS.rightlim(psi) <= length(psi) + 1

@testset "other ortho-limit setters" begin
    Random.seed!(2468)
    s = siteinds(2, 10)

    @testset "ttruncate!, site_range = $sr" for sr in (1:10, 3:7)
        psi = random_mps(ComplexF64, s; linkdims = 6)
        ttruncate!(psi; site_range = sr, maxdim = 3)
        @test ortho_residual(psi) < 1e-10
        @test norm_consistent(psi)
    end

    @testset "tcontract densitymatrix" begin
        psi = random_mps(ComplexF64, s; linkdims = 4)
        A   = random_mpo(s) + im * random_mpo(s)
        out = tapply(ITransverse.Algorithm("densitymatrix"), A, psi; cutoff = 0.0, maxdim = 6)
        out = out isa Tuple ? first(out) : out
        @test ortho_residual(out) < 1e-10
        @test norm_consistent(out)
    end

    @testset "gen_orthogonalize: limits only where the form is unitary" begin
        # complex: bilinear isometries are not unitary ones -> limits must stay unknown
        g = gen_orthogonalize(random_mps(ComplexF64, s; linkdims = 4), 5)
        @test !isortho(g)
        @test norm_consistent(g)
        # real: the two forms coincide -> the centre is recorded and is truthful
        gr = gen_orthogonalize(random_mps(Float64, s; linkdims = 4), 5)
        @test isortho(gr) && orthocenter(gr) == 5
        @test ortho_residual(gr) < 1e-10
        @test norm_consistent(gr)
    end

    @testset "normbyfactor does not touch its input" begin
        psi = random_mps(ComplexF64, s; linkdims = 4)
        ref = deepcopy(psi)
        out = normbyfactor(psi, 7.0)
        @test all(norm(psi[j] - ref[j]) == 0 for j in eachindex(psi))
        @test ITensorMPS.ortho_lims(psi) == ITensorMPS.ortho_lims(ref)
        @test inner(out, out) ≈ inner(ref, ref) / 49
        @test norm_consistent(out)
    end

    @testset "combine_and_fold leaves valid limits after dropping sites" begin
        tp  = ising_tp()
        bfw = FwtMPOBlocks(tp)
        ts  = siteinds("S=1/2", 8)
        fwmps = fw_tMPS(bfw, ts; tr = [1, 1])
        fold, _, _ = ITransverse.combine_and_fold(fwmps, fwmps; fold_op = [1 0; 0 1],
                                                  fold_init_state = [1, 0, 0, 1], dag_W2 = true)
        @test valid_lims(fold)
        @test !isortho(fold)
        f2 = orthogonalize(fold, 1)          # used to index past the end of the chain
        @test norm_consistent(f2)
    end
end
