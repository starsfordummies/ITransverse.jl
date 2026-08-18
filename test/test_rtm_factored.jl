"""
The factored RTM SVD (`factored=true`, the default; see
`truncation_sweeps/rtm_svd.jl`) must return exactly what building `rho` and
SVDing it densely returns: same singular values, same singular vectors up to the
usual gauge, same pre-truncation overlap. It only skips work the rank bound of
the factored RTM says is redundant.

Where no side is rank deficient the kernel takes the dense route, so the two
settings must then agree *bitwise* -- that is the folded-Ising two-tMPO case,
where D = d = 4.
"""

using ITensors, ITensorMPS, ITransverse
using LinearAlgebra
using Test

""" Random complex MPO of the given bond dimension (`random_mpo` is real, m==1). """
function _crandom_mpo(ss, chi)
    N  = length(ss)
    ls = [Index(chi, "Link,l=$j") for j in 1:N-1]
    A  = MPO(N)
    for j in 1:N
        is = j == 1 ? (ss[1]', ss[1], ls[1]) :
             j == N ? (ls[N-1], ss[N]', ss[N]) :
                      (ls[j-1], ss[j]', ss[j], ls[j])
        t = random_itensor(ComplexF64, is...)
        A[j] = t / norm(t)
    end
    return A
end

_maxreldiff(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(a)), eps())

""" Dense-vs-factored agreement for one `TruncLR` pair. """
function _agree(dense, fact; tol = 1e-10)
    @test _maxreldiff(dense.sv, fact.sv) < tol
    @test abs(1 - abs(fidelity(dense.L, fact.L))) < tol
    @test abs(1 - abs(fidelity(dense.R, fact.R))) < tol
    a1 = overlap_noconj(dense.L, dense.R)
    a2 = overlap_noconj(fact.L, fact.R)
    @test abs(a1 - a2) / abs(a1) < tol
    @test dense.ov_before ≈ fact.ov_before
end

@testset "factored RTM SVD: random MPS/MPO" begin
    ss = siteinds(4, 14)
    for (chiL, chiR, chiA) in ((24, 24, 4), (24, 16, 6), (32, 32, 2))
        ψL = random_mps(ComplexF64, ss, linkdims=chiL)
        ψR = random_mps(ComplexF64, ss, linkdims=chiR)
        AL = _crandom_mpo(ss, chiA)
        AR = _crandom_mpo(ss, chiA)
        tk = (alg="RTM", cutoff=1e-14, maxdim=64, mindim=1)
        for dir in (:right, :left)
            _agree(trapply(ψL, AR, ψR; direction=dir, factored=false, tk...),
                   trapply(ψL, AR, ψR; direction=dir, factored=true,  tk...))
            _agree(tlapply(ψL, AL, ψR; direction=dir, factored=false, tk...),
                   tlapply(ψL, AL, ψR; direction=dir, factored=true,  tk...))
            _agree(tlrapply(ψL, AL, AR, ψR; direction=dir, factored=false, tk...),
                   tlrapply(ψL, AL, AR, ψR; direction=dir, factored=true,  tk...))
        end
    end
end

@testset "factored RTM SVD: folded Ising columns" begin
    mp = IsingParams(1.0, 0.95, 1.4)
    tp = tMPOParams(mp; dt=0.1, scheme=Murg(), init_state=[1, 0])
    b  = FoldtMPOBlocks(tp)
    ts = siteinds(4, 24)
    col   = folded_tMPO(b, ts)
    colop = folded_tMPO(b, ts; fold_op=[1, 0, 0, -1])

    # A pair of environments with a non-trivial bond dimension to truncate: grow
    # each side independently with the density-matrix apply, so the RTM sweeps
    # under test play no part in preparing their own input.
    ψR = folded_tMPS(b, ts; LR=:right)
    ψL = folded_tMPS(b, ts; LR=:left)
    for _ in 1:4
        ψR = first(tapply(ITransverse.Algorithm("densitymatrix"), col, ψR; cutoff=1e-14, maxdim=32))
        ψL = first(tapplys(ITransverse.Algorithm("densitymatrix"), col, ψL; cutoff=1e-14, maxdim=32))
    end
    # guard against the checks below going vacuous on product-state environments
    @test maxlinkdim(ψR) > 4
    @test maxlinkdim(ψL) > 4

    for dir in (:right, :left)
        _agree(trapply(ψL, col, ψR; direction=dir, factored=false, alg="RTM", cutoff=1e-14, maxdim=32, mindim=1),
               trapply(ψL, col, ψR; direction=dir, factored=true,  alg="RTM", cutoff=1e-14, maxdim=32, mindim=1))
        _agree(tlapply(ψL, col, ψR; direction=dir, factored=false, alg="RTM", cutoff=1e-14, maxdim=32, mindim=1),
               tlapply(ψL, col, ψR; direction=dir, factored=true,  alg="RTM", cutoff=1e-14, maxdim=32, mindim=1))
        _agree(tlrapply(ψL, col, colop, ψR; direction=dir, factored=false, alg="RTM", cutoff=1e-14, maxdim=32, mindim=1),
               tlrapply(ψL, col, colop, ψR; direction=dir, factored=true,  alg="RTM", cutoff=1e-14, maxdim=32, mindim=1))
    end
end

@testset "factored RTM SVD: QN (block-sparse)" begin
    # Z2 (Sz parity) conserving unfolded transverse machinery, as in test_qn_ising.jl
    mp = IsingParams(1.0, 0.7, 0.0)
    up = ComplexF64[1, 0]
    ss = siteinds("S=1/2", 3; conserve_szparity=true)
    b  = FwtMPOBlocks(build_Ut(ss, Murg(), mp; dt=0.1); init_state=up)
    ts = [sim(b.iP, tags="Site,time,t=$i") for i in 1:20]
    col = fw_tMPO(b, ts; tr=up)
    ψL  = fw_tMPS(b, ts; LR=:left, tr=up)
    ψR  = fw_tMPS(b, ts; LR=:right, tr=up)
    @test hasqns(col)

    for _ in 1:5
        ψR = first(tapply(ITransverse.Algorithm("densitymatrix"), col, ψR; cutoff=1e-14, maxdim=32))
        ψL = first(tapplys(ITransverse.Algorithm("densitymatrix"), col, ψL; cutoff=1e-14, maxdim=32))
    end
    @test maxlinkdim(ψR) > 4
    @test hasqns(ψR[2]) && hasqns(ψL[2])

    tk = (alg="RTM", cutoff=1e-14, maxdim=32, mindim=1)
    for dir in (:right, :left)
        # `qr` works on block-sparse tensors, so :always is exact there too
        _agree(trapply(ψL, col, ψR; direction=dir, factored=false,   tk...),
               trapply(ψL, col, ψR; direction=dir, factored=:always, tk...))
        _agree(tlapply(ψL, col, ψR; direction=dir, factored=false,   tk...),
               tlapply(ψL, col, ψR; direction=dir, factored=:always, tk...))
        _agree(tlrapply(ψL, col, col, ψR; direction=dir, factored=false,   tk...),
               tlrapply(ψL, col, col, ψR; direction=dir, factored=:always, tk...))

        # and the symmetry survives the factored route
        @test hasqns(trapply(ψL, col, ψR; direction=dir, factored=:always, tk...).R[2])

        # `factored=true` is "factorize where it pays", which for QN means the
        # dense route -- so it must be *identical*, not merely close
        auto  = trapply(ψL, col, ψR; direction=dir, factored=true,  tk...)
        dense = trapply(ψL, col, ψR; direction=dir, factored=false, tk...)
        @test auto.sv == dense.sv
    end
end

@testset "factored RTM SVD: svd_rtm kernel" begin
    # rho[a,b] = R[a,i] E[i,j] L[b,j] with a deficient internal leg on the L side
    D, chi, d, k = 4, 8, 4, 3
    sa, ua = Index(D, "Site,a"), Index(chi, "Link,ua")
    sb, ub = Index(D, "Site,b"), Index(chi, "Link,ub")
    i1, i2 = Index(chi, "Link,i1"), Index(d, "Link,i2")
    j1     = Index(k, "Link,j1")

    R = random_itensor(ComplexF64, sa, ua, i1, i2)
    E = random_itensor(ComplexF64, i1, i2, j1)
    L = random_itensor(ComplexF64, sb, ub, j1)

    tk = (cutoff=0.0, maxdim=D*chi, mindim=1, lefttags="Link,u", righttags="Link,v")
    U1, S1, V1, _ = svd_rtm(E, R, L, IndexSet(sa, ua); factored=false, tk...)
    U2, S2, V2, _ = svd_rtm(E, R, L, IndexSet(sa, ua); factored=true,  tk...)
    U3, S3, V3, _ = svd_rtm(E, R, L, IndexSet(sa, ua); factored=:always, tk...)
    # no QNs here, so `true` and `:always` take the same route (fresh Index ids
    # each call, so compare the spectra, not the ITensors)
    @test diag(Array(S2, inds(S2))) == diag(Array(S3, inds(S3)))

    # rank is capped by the small internal leg, and the factored route must
    # reproduce rho itself, not just its spectrum
    @test dim(commonind(S2, U2)) <= k
    # with cutoff=0 the dense route also returns the ~1e-16 tail the factored
    # route never forms, so compare the leading values and check the tail is noise
    s1 = sort(diag(Array(S1, inds(S1))); rev=true)
    s2 = sort(diag(Array(S2, inds(S2))); rev=true)
    n  = min(length(s1), length(s2))
    @test _maxreldiff(s1[1:n], s2[1:n]) < 1e-12
    @test maximum(s1[n+1:end]; init=0.0) < 1e-10 * s1[1]
    rho = E * R * L
    @test norm(U2 * S2 * V2 - rho) / norm(rho) < 1e-12
    @test norm(U1 * S1 * V1 - rho) / norm(rho) < 1e-12
end
