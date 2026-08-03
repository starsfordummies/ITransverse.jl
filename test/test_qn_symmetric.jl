using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra
using Logging

using ITransverse: transpose_arrows, transpose_matrix, blockwise_sqrt, blockwise_invsqrt,
    blockwise_matfun, symm_svd, symm_oeig, spectrum_vector

# The "symmetric" (RTM) machinery with QNs. These algorithms contract a state with a
# *transposed* copy of itself; with QNs a transpose reverses the arrows, which is what
# `transpose_arrows` supplies. See test_qn_ising.jl for the builders.

const MP = IsingParams(1.0, 1.0, 0.0)   # critical, hpar=0 so SzParity is conserved
const DT = 0.1
const UP = ComplexF64[1, 0]

""" tMPO + right/left tMPS, with or without QNs """
function setup(Nt::Int; qns::Bool)
    ss = siteinds("S=1/2", 3; conserve_szparity=qns)
    U3 = with_logger(NullLogger()) do
        build_Ut(ss, Murg(), MP; dt=DT)
    end
    b = with_logger(NullLogger()) do
        FwtMPOBlocks(U3; init_state=UP)
    end
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:Nt]
    return (fw_tMPO(b, ts; tr=UP),
            fw_tMPS(b, ts; LR=:left, tr=UP),
            fw_tMPS(b, ts; LR=:right, tr=UP))
end

""" a complex-symmetric environment, exactly as the RTM sweeps build it """
function rtm_env(R::MPS)
    Rc = orthogonalize(R, 1)
    A = Rc[1]
    sA = only(siteinds(Rc, 1))
    return ITransverse.symmetrize(A * noprime(prime(transpose_arrows(A)), prime(dag(sA))))
end


@testset "arrow-aware transpose and block-wise matrix functions" begin
    T, L, R = setup(6; qns=true)
    env = rtm_env(R)
    i, j = inds(env)

    @test hasqns(env)
    @test nnzblocks(env) == 2
    # the RTM environment really is complex-symmetric under the arrow-aware transpose,
    # which a plain `swapinds` cannot even express here (the legs are dual)
    @test norm(env - transpose_matrix(env, i, j)) / norm(env) < 1e-12
    @test_throws ErrorException norm(env - swapinds(env, (i,), (j,)))

    # block-wise matrix functions reproduce the dense ones
    envd = dense(env)
    id_, jd = inds(envd)
    @test norm(dense(blockwise_sqrt(env)) - sqrt(envd)) / norm(envd) < 1e-10
    s = blockwise_sqrt(env)
    @test norm(dense(blockwise_matfun(M -> M * M, s)) - envd) / norm(envd) < 1e-10
    @test norm(blockwise_invsqrt(env) - blockwise_matfun(M -> M^-0.5, env)) < 1e-12

    # and they are exact no-ops on the plain (non-QN) path
    @test transpose_matrix(envd, id_, jd) ≈ swapinds(envd, (id_,), (jd,))
    @test transpose_arrows(envd) ≈ envd
end


@testset "complex-symmetric SVD with QNs" begin
    T, L, R = setup(6; qns=true)
    env = rtm_env(R)
    iL = ind(env, 1)

    F = symm_svd(env, iL; cutoff=1e-14)
    @test hasqns(F.U)
    @test nnzblocks(F.S) == 2
    # a = U S transpose(U)
    @test norm(F.U * F.S * F.V - env) / norm(env) < 1e-12

    # identical decomposition through the same code path without QNs
    envd = dense(env)
    Fd = symm_svd(envd, ind(envd, 1); cutoff=1e-14)
    @test norm(Fd.U * Fd.S * Fd.V - envd) / norm(envd) < 1e-12
    @test dim(F.u) == dim(Fd.u)
    @test sort(spectrum_vector(F.S); rev=true) ≈ sort(spectrum_vector(Fd.S); rev=true)
end


@testset "symmetric truncation sweeps with QNs" begin
    T, L, R = setup(6; qns=true)
    TR = applyn(T, R)

    for f in (ITransverse.truncate_sweep_sym,)
        out, sv = f(TR; cutoff=1e-12, maxdim=32)
        @test hasqns(out)
        @test length(out) == length(TR)
    end

    out2, _ = ITransverse.truncate_sweep_sym_rtm!(copy(TR); maxdim=32)
    @test hasqns(out2)

    # tapply with the RTM-symmetric algorithm: must agree with the untruncated result
    exact = overlap_noconj(L, applyn(T, R))
    for alg in ("naive", "densitymatrix", "RTMsym")
        out, _ = tapply(T, R; alg, cutoff=1e-14, maxdim=64)
        @test hasqns(out)
        @test isapprox(overlap_noconj(L, out), exact; rtol=1e-6)
    end
end


@testset "QN and non-QN symmetric power method agree" begin
    res = Dict{Bool,Any}()
    for qns in (false, true)
        T, L, R = setup(6; qns)
        pmp = PMParams(; truncp=(; cutoff=1e-14, maxdim=64, alg="RTMsym"), itermax=40,
                       eps_converged=1e-11, opt_method=:sym, normalization="norm")
        psi, _ = with_logger(NullLogger()) do
            powermethod_sym(R, T, pmp)
        end
        lead = overlap_noconj(psi, applyn(T, psi)) / overlap_noconj(psi, psi)
        res[qns] = (; psi, lead)
    end
    @test hasqns(res[true].psi)
    @test !hasqns(res[false].psi)
    @test isapprox(abs(res[true].lead), abs(res[false].lead); rtol=1e-5)
end


@testset "QN: the eigen-based symmetric routines still fail loudly" begin
    T, L, R = setup(4; qns=true)
    env = rtm_env(R)

    # symm_oeig (complex-symmetric *eigen*) has no block-sparse implementation yet: it must
    # error rather than silently densify. Everything SVD-based above works.
    @test_throws ErrorException symm_oeig(env, ind(env, 1))
    @test_throws ErrorException gen_canonical(R, length(R))
    @test_throws ErrorException ITransverse.truncate_sweep_sym(applyn(T, R);
                                    cutoff=1e-12, maxdim=16, use_eig=true)
end
