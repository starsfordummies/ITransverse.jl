using ITensors, ITensorMPS, ITransverse
using LinearAlgebra
using Logging
using NDTensors
using Random
using Test

using ITransverse: symm_svd, symm_svd_legacy, symm_oeig, symm_oeig_legacy,
    sym_unitary_sqrt, degenerate_blocks, isapproxdiag, symmetrize

randsym(n) = (A = randn(ComplexF64, n, n); (A + transpose(A)) / 2)
recon_err(F, M) = norm(F.U * Diagonal(F.S) * transpose(F.U) - M) / norm(M)

""" QN matrix-like ITensor, block diagonal, complex symmetric, with `evs[k]` as the
eigenvalues of block `k`. `±σ` pairs there make `z` have eigenvalues of exactly -1. """
function qn_sym_env(evs; seed)
    Random.seed!(seed)
    d = length(first(evs))
    i = Index(QN("P", 0, 2) => d, QN("P", 1, 2) => d; dir=ITensors.Out, tags="l")
    at = ITensors.tensor(random_itensor(ComplexF64, i, dag(i)'))
    for (k, bl) in enumerate(nzblocks(at))
        Q = Matrix(qr(randn(d, d)).Q)
        NDTensors.blockview(at, bl) .= ComplexF64.(Q * Diagonal(evs[k]) * Q')
    end
    return symmetrize(itensor(at)), i
end


""" complex-orthogonal O (transpose(O)*O = I); larger `scale` makes it more
ill-conditioned, i.e. pushes M towards a defective matrix. """
cplx_orth(n, scale) = exp(scale * (A -> (A - transpose(A)) / 2)(randn(ComplexF64, n, n)))


@testset "Takagi (complex-symmetric) SVD" begin

    # ---------------------------------------------------------------------------------
    @testset "new and legacy agree on sane inputs" begin
        Random.seed!(4321)
        for n in (2, 9, 40, 129)
            M = randsym(n)
            Fn, specn = symm_svd(M)
            Fl, specl = symm_svd_legacy(M)
            @test Fn.S ≈ Fl.S
            @test Fn.U ≈ Fl.U                       # same phases, not merely the same subspace
            @test recon_err(Fn, M) < 1e-12
        end

        # ... and with truncation, where the spectrum is cut before the fix-up
        M = 1234 * randsym(60)
        for cutoff in (1e-14, 1e-8), maxdim in (60, 17)
            Fn, sn = symm_svd(M; cutoff, maxdim)
            Fl, sl = symm_svd_legacy(M; cutoff, maxdim)
            @test Fn.S ≈ Fl.S
            @test Fn.U ≈ Fl.U
            @test norm(Fn.U * Diagonal(Fn.S) * transpose(Fn.U) - M) / norm(M) <
                  max(sqrt(cutoff), 1e-12) + (maxdim < 60) * 1.0
        end

        # ITensor path, dense and with QNs, on a well-conditioned environment
        Random.seed!(77)
        env, i = qn_sym_env(([3.0, 1.7, 0.9, 0.4], [2.2, 1.1, 0.6, 0.2]); seed=12)
        for a in (env, dense(env))
            ia = ind(a, 1)
            Fn = symm_svd(a, ia; cutoff=1e-14)
            Fl = symm_svd_legacy(a, ia; cutoff=1e-14)
            @test norm(Fn.U * Fn.S * Fn.V - a) / norm(a) < 1e-12
            # the two calls mint their own link indices, so relabel before comparing
            Un = replaceind(Fn.U, Fn.u => Fl.u)
            Sn = replaceinds(Fn.S, (Fn.u, Fn.v) => (Fl.u, Fl.v))
            @test norm(Un - Fl.U) / norm(Fl.U) < 1e-10
            @test norm(Sn - Fl.S) / norm(Fl.S) < 1e-12
        end
    end

    # ---------------------------------------------------------------------------------
    @testset "sym_unitary_sqrt on the branch cut" begin
        Random.seed!(2)
        k = 12
        Q = Matrix(qr(randn(k, k)).Q)
        # eigenvalue pairs exp(±im(π-δ)) sit either side of the cut of the principal sqrt;
        # δ = 0 is the real symmetric indefinite case, where they are exactly -1
        for δ in (1e-1, 1e-4, 1e-8, 0.0)
            z = Q * Diagonal(cis.([isodd(j) ? π - δ : -(π - δ) for j in 1:k])) * transpose(Q)
            z = (z + transpose(z)) / 2
            w = sym_unitary_sqrt(z)
            @test norm(w * transpose(w) - z) < 1e-12        # it *is* a square root
            @test norm(w'w - I) < 1e-12                     # and it is unitary
        end

        # diagonal fast path and the 1x1 case
        ph = cis.([0.3, -2.9, π, -π])
        @test sym_unitary_sqrt(Matrix(Diagonal(ph))) ≈ Diagonal(sqrt.(ph))
        @test sym_unitary_sqrt(fill(-1.0 + 0im, 1, 1))[1] ≈ im
    end

    # ---------------------------------------------------------------------------------
    @testset "degenerate singular values" begin
        @test degenerate_blocks([4.0, 4.0, 2.0, 1.0, 1.0, 1.0]) == [1:2, 3:3, 4:6]
        @test degenerate_blocks([4.0, 4.0 - 1e-9, 2.0]) == [1:2, 3:3]
        @test degenerate_blocks(Float64[]) == UnitRange{Int}[]

        # real symmetric and indefinite: |σ| is degenerate and z has eigenvalues -1
        Random.seed!(99)
        for trial in 1:25
            Q = Matrix(qr(randn(24, 24)).Q)
            M = ComplexF64.(Q * Diagonal(repeat([3.0, -3.0, 2.0, -2.0, 1.0, -1.0], 4)) * Q')
            F, _ = symm_svd(M)
            @test recon_err(F, M) < 1e-10
            @test norm(F.U'F.U - I) < 1e-10
        end

        # complex, with degenerate blocks of prescribed multiplicity
        Random.seed!(5)
        for mult in (2, 5, 20)
            U = Matrix(qr(randn(ComplexF64, 20, 20)).Q)
            s = sort(rand(20) .+ 0.5; rev=true)
            for b in 1:mult:(20 - mult + 1)
                s[b:(b + mult - 1)] .= s[b]
            end
            M = U * Diagonal(s) * transpose(U)
            F, _ = symm_svd(M)
            @test recon_err(F, M) < 1e-10
            @test F.S ≈ s
        end

        # a real (not merely real-valued-complex) input must not throw
        Q = Matrix(qr(randn(12, 12)).Q)
        Mr = Q * Diagonal([3.0, -3.0, 2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.7, -0.7, 0.2, -0.2]) * Q'
        F, _ = symm_svd(Mr)
        @test recon_err(F, ComplexF64.(Mr)) < 1e-10
    end

    # ---------------------------------------------------------------------------------
    @testset "QN blocks with degenerate singular values" begin
        for seed in 1:15
            env, i = qn_sym_env(([3.0, -3.0, 2.0, -2.0], [1.5, -1.5, 1.0, -1.0]); seed)
            envd = dense(env)
            F = symm_svd(env, i; cutoff=1e-14)
            Fd = symm_svd(envd, ind(envd, 1); cutoff=1e-14)
            @test hasqns(F.U)
            @test nnzblocks(F.S) == 2
            @test norm(F.U * F.S * F.V - env) / norm(env) < 1e-10
            @test norm(Fd.U * Fd.S * Fd.V - envd) / norm(envd) < 1e-10
            @test sort(ITransverse.spectrum_vector(F.S); rev=true) ≈
                  sort(ITransverse.spectrum_vector(Fd.S); rev=true)
        end

        # a real-valued QN environment: the phases are complex even though z is not, so the
        # block-wise square root has to be written into a complex copy
        env, i = qn_sym_env(([3.0, -3.0, 2.0, -2.0], [1.5, -1.5, 1.0, -1.0]); seed=3)
        envr = real(env)
        @test eltype(envr) == Float64
        Fr = symm_svd(envr, i; cutoff=1e-14)
        @test norm(Fr.U * Fr.S * Fr.V - envr) / norm(envr) < 1e-10
    end

    # ---------------------------------------------------------------------------------
    @testset "symm_factorization reproduces the tensor, scale included" begin
        # the fix-up inside `symm_factorization` carries the singular values as well as the
        # phases, so normalising it (as `sym_unitary_sqrt` does) would silently rescale the
        # two halves - exactly what test_folded_tmpo / test_time_evol_ising caught
        Random.seed!(31)
        for evs in ([3.0, 1.7, 0.9, 0.4], [3.0, -3.0, 2.0, -2.0])
            Q = Matrix(qr(randn(4, 4)).Q)
            M = ComplexF64.(Q * Diagonal(evs) * Q')
            i1, i2 = Index(4, "i1"), Index(4, "i2")
            a = ITensor(M, i1, i2)
            uuL, uuR = ITransverse.symm_factorization(a, (i1,); cutoff=1e-14)
            @test norm(uuL * uuR - a) / norm(a) < 1e-10
        end
    end

    # ---------------------------------------------------------------------------------
    @testset "near-degenerate singular values" begin
        # the worst case sits at the grouping crossover: below it the block keeps a spread of
        # g, above it an off-diagonal element of size ~eps/g is dropped. Neither is exact, so
        # this pins the size of the compromise rather than asserting machine precision
        for g in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12)
            worst = 0.0
            for t in 1:10
                Random.seed!(t)
                U = Matrix(qr(randn(ComplexF64, 40, 40)).Q)
                s = sort(rand(40) .+ 0.5; rev=true)
                for b in 1:2:39
                    s[b + 1] = s[b] * (1 - g)
                end
                s = sort(s; rev=true)
                M = U * Diagonal(s) * transpose(U)
                F, _ = symm_svd(M)
                worst = max(worst, recon_err(F, M))
            end
            @test worst < 1e-7
        end
    end

    # ---------------------------------------------------------------------------------
    @testset "isapproxdiag is not fooled by cancellation" begin
        # z = u' * conj(v) is exactly diagonal here, but ||z||² and ||diag(z)||² are both n:
        # taking their difference leaves only rounding noise, whose square root grows like
        # sqrt(n*eps) and crosses the 1e-8 tolerance around n = 400
        Random.seed!(7)
        for n in (16, 128, 512, 900)
            u, s, v = svd(randsym(n))
            z = u' * conj(v)
            @test norm(z - Diagonal(diag(z))) < 1e-10       # it really is diagonal
            @test isapproxdiag(z)                           # and we detect that
            @test !isapproxdiag(u)                          # a generic unitary is not
        end
    end
end


@testset "complex-orthogonal (symmetric) eigendecomposition" begin

    @testset "new and legacy agree on sane inputs" begin
        for n in (6, 20, 60)
            Random.seed!(n)
            M = randsym(n)
            Fn, specn = symm_oeig(M)
            Fl, specl = symm_oeig_legacy(M)
            @test Fn.values == Fl.values
            @test Fn.vectors ≈ Fl.vectors
            @test norm(Fn.vectors * Diagonal(Fn.values) * transpose(Fn.vectors) - M) / norm(M) < 1e-12
            @test norm(transpose(Fn.vectors) * Fn.vectors - I) < 1e-10
        end

        # with truncation
        Random.seed!(5)
        M = 1234 * randsym(40)
        for cutoff in (1e-12, 1e-6)
            Fn, _ = symm_oeig(M; cutoff)
            Fl, _ = symm_oeig_legacy(M; cutoff)
            @test Fn.values == Fl.values
            @test Fn.vectors ≈ Fl.vectors
        end
    end

    @testset "degenerate eigenvalues (Z is genuinely non-diagonal)" begin
        for scale in (0.5, 1.5)
            for t in 1:10
                Random.seed!(t)
                O0 = cplx_orth(8, scale)
                M = O0 * Diagonal(ComplexF64[3, 3, 2, 2, 1, 1, 0.4, 0.4]) * transpose(O0)
                F, _ = with_logger(NullLogger()) do
                    symm_oeig(M)
                end
                @test norm(F.vectors * Diagonal(F.values) * transpose(F.vectors) - M) / norm(M) < 1e-10
                @test norm(transpose(F.vectors) * F.vectors - I) < 1e-10
            end
        end
    end

    @testset "ill-conditioned bases are reported" begin
        # O is complex-orthogonal, so cond(O) = σmax(O)^2 and it diverges as M approaches a
        # defective matrix. That cannot be fixed, only reported.
        Random.seed!(11)
        evs = ComplexF64[3, 2.5, 2, 1.5, 1, 0.7, 0.4, 0.2]
        Mgood = (O0 = cplx_orth(8, 1.5); O0 * Diagonal(evs) * transpose(O0))
        Random.seed!(11)
        Mbad = (O0 = cplx_orth(8, 4.5); O0 * Diagonal(evs) * transpose(O0))

        @test_logs min_level=Logging.Warn symm_oeig(Mgood)
        @test_logs (:warn, r"ill-conditioned") match_mode=:any symm_oeig(Mbad)

        # and the warning tracks a real loss of accuracy
        Fg, _ = symm_oeig(Mgood)
        Fb, _ = with_logger(NullLogger()) do
            symm_oeig(Mbad)
        end
        eg = norm(Fg.vectors * Diagonal(Fg.values) * transpose(Fg.vectors) - Mgood) / norm(Mgood)
        eb = norm(Fb.vectors * Diagonal(Fb.values) * transpose(Fb.vectors) - Mbad) / norm(Mbad)
        @test eg < 1e-11
        @test eb > 1e-9
    end
end


@testset "Cutoff convention" begin

    @testset "cutoff is linear in the singular values" begin
        # A reduced transition matrix is two layers of the state, so its singular values are
        # already probabilities rather than amplitudes and a cutoff `c` has to discard a
        # fraction `c` of `sum(s)`. `symm_svd` used to inherit ITensors' `svd` convention and
        # threshold `sum(s^2)` instead, which discards `sqrt(c)`: on `test_loschmidt.jl` that
        # collapsed the RTMsym/SVD bond dimension to 6 where the eig and densitymatrix routes
        # reach 19-20 at the same nominal cutoff, and the power method stopped converging.
        Random.seed!(20250921)
        n = 60
        s_full = 10.0 .^ range(0, -14; length=n)
        U = Matrix(qr(randn(ComplexF64, n, n)).Q)
        M = U * Diagonal(s_full) * transpose(U)
        i, j = Index(n, "i"), Index(n, "j")
        T = ITensor(M, i, j)

        for c in (1e-4, 1e-8, 1e-12)
            F, spec = symm_svd(copy(M); cutoff=c)
            discarded = (sum(s_full) - sum(F.S)) / sum(s_full)

            @test discarded <= c              # the definition; was ~sqrt(c) before
            @test discarded > c / 100         # and it is not keeping far more than asked
            @test recon_err(F, M) <= 10c      # norm error ~ c, not ~ sqrt(c)
            @test spec.truncerr <= c

            # the ITensor path takes the same cut, through `svd_trunc_values`
            @test dim(symm_svd(T, i; cutoff=c).u) == length(F.S)
        end

        # the ITensors convention is still reachable, and is what the difference looks like
        Fsq, _ = ITransverse.truncated_svd(copy(M); cutoff=1e-12, cutoff_on=:squares)
        @test length(Fsq.S) < length(symm_svd(copy(M); cutoff=1e-12)[1].S)
    end

    @testset "symm_svd and symm_oeig agree on what a cutoff means" begin
        # One `cutoff` has to mean one thing across RTMsym/SVD, RTMsym/EIG and densitymatrix.
        # `M` is real symmetric positive here so that its singular values and its eigenvalues
        # coincide and the two routes are comparable exactly.
        Random.seed!(4)
        n = 50
        p = 10.0 .^ range(0, -13; length=n)
        Q = Matrix(qr(randn(n, n)).Q)
        M = Q * Diagonal(p) * transpose(Q)

        for c in (1e-3, 1e-6, 1e-9, 1e-12)
            ksvd = length(symm_svd(copy(M); cutoff=c)[1].S)
            keig = length(symm_oeig(copy(M); cutoff=c)[1].values)
            @test abs(ksvd - keig) <= 1
        end
    end
end
