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
