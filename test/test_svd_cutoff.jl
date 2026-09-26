"""
`svd_cutoff` (used by the RTM / RTMsym / naiveRTM truncations) with its three
`cutoff_on` rules:
  :values_bench keeps exactly the count of the linear rule (discarded sum(S) <= cutoff*sum(S)),
  :squares      is ITensors' own svd,
  :values       is ITensors' svd at cutoff^2.
"""

using ITensors, ITensorMPS, ITransverse
using LinearAlgebra, Random
using Test

@testset "svd_cutoff" begin
    Random.seed!(7)
    i, j = Index(60, "i"), Index(50, "j")
    # known spectrum: two decays, so the linear and squares rules differ
    s = vcat(10.0 .^ range(0, -3; length=20), 10.0 .^ range(-3.2, -9; length=30))
    Q1 = Matrix(qr(randn(ComplexF64, 60, 50)).Q)[:, 1:50]
    Q2 = Matrix(qr(randn(ComplexF64, 50, 50)).Q)
    A  = ITensor(Q1 * Diagonal(s) * Q2', i, j)

    linear_count(c) = findfirst(k -> sum(s[k+1:end]) <= c * sum(s), 1:length(s))

    for c in (1e-2, 1e-4, 1e-6)
        Fb, specb = svd_cutoff(A, i; cutoff_on=:values_bench, cutoff=c)
        Fv, _     = svd_cutoff(A, i; cutoff_on=:values, cutoff=c)
        Fs, _     = svd_cutoff(A, i; cutoff_on=:squares, cutoff=c)

        @test dim(Fb.u) == linear_count(c)
        @test dim(Fs.u) == dim(svd(A, i; cutoff=c).u)
        @test dim(Fv.u) == dim(svd(A, i; cutoff=c^2).u)

        for F in (Fb, Fv, Fs)
            @test storage(F.S) isa NDTensors.Diag
            m = dim(F.u)
            # the kept part of an exact SVD: error = the discarded singular values
            @test norm(A - F.U * F.S * F.V) ≈ norm(s[m+1:end]) rtol=1e-6
        end
    end

    # nothing / 0 keep everything, a positive cutoff below eps is clamped to eps
    @testset "linear_cutoff" begin
        @test ITransverse.linear_cutoff(nothing) === nothing
        @test ITransverse.linear_cutoff(0.0) === nothing
        @test ITransverse.linear_cutoff(0) === nothing
        @test ITransverse.linear_cutoff(1e-20) == eps()
        @test ITransverse.linear_cutoff(1e-12) == 1e-12

        for m in (:values, :values_bench), c in (nothing, 0.0)
            F, _ = svd_cutoff(A, i; cutoff_on=m, cutoff=c)
            @test dim(F.u) == length(s)
        end
        for m in (:values, :values_bench)
            Ftiny, _ = svd_cutoff(A, i; cutoff_on=m, cutoff=1e-30)
            Feps, _  = svd_cutoff(A, i; cutoff_on=m, cutoff=eps())
            @test dim(Ftiny.u) == dim(Feps.u)
        end
    end

    @test_throws ArgumentError svd_cutoff(A, i; cutoff_on=:bogus, cutoff=1e-4)
end
