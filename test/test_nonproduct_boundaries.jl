using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra

using Random

using ITransverse: boundary_tensor, close_boundary, fold_boundary, to_boundary,
    check_boundary, boundary_bond_ind, boundary_phys_ind, n_boundary_sites,
    is_product_boundary, boundary_linkdim

# random boundary states below; seed so the tolerances mean something run to run
Random.seed!(20250803)

# Non-product (bond dimension > 1) initial/final states for the transverse builders.
# Reference values always come from a plain real-space contraction of a finite TN.

""" Translation-invariant MPS of `length(ss)` sites out of the bulk tensor `A` (legs
`sphys`, `il`, `ir`) and the edge vectors `vL`, `vR`. """
function ti_mps(A::ITensor, sphys::Index, il::Index, ir::Index, vL, vR, ss)
    LL = length(ss)
    chi = dim(ir)
    bonds = [Index(chi, "Link,a=$j") for j in 1:(LL-1)]
    Ai = ITensor[replaceinds(A, (sphys, ir), (ss[1], bonds[1])) * ITensor(vL, il)]
    for j in 2:(LL-1)
        push!(Ai, replaceinds(A, (sphys, il, ir), (ss[j], bonds[j-1], bonds[j])))
    end
    push!(Ai, replaceinds(A, (sphys, il), (ss[LL], bonds[LL-1])) * ITensor(vR, ir))
    return MPS(Ai)
end

""" <phi|psi> *without* conjugating either side """
function overlap_mps_noconj(phi::MPS, psi::MPS)
    O = ITensors.OneITensor()
    for j in eachindex(psi)
        O = O * psi[j] * phi[j]
    end
    return scalar(O)
end

""" Contract the transverse network of `L` columns: 2 edge tMPS + (L-2) bulk tMPO """
function contract_columns(ll::TMPSorMPS, mpo::MPO, rr::TMPSorMPS, L::Int)
    r = rr
    for _ in 1:(L-2)
        # a column is applied from the side the vector is on; `overlap_noconj` then checks
        # that we are pairing a left with a right
        r = apply_column(mpo, r)
    end
    return overlap_noconj(ll, r)
end


@testset "boundary state conventions" begin
    chi = 3
    sp = Index(2, "S=1/2")
    il = Index(chi, "il")
    ir = Index(chi, "ir")
    A = random_itensor(ComplexF64, sp, il, ir)

    bl = boundary_tensor(A; phys=sp, left=il, right=ir)
    @test ndims(bl) == 3
    @test boundary_phys_ind(bl) == only(inds(bl, "Site"))
    @test dim(boundary_phys_ind(bl)) == 2
    @test !is_product_boundary(bl)
    @test n_boundary_sites(bl) == 1
    @test boundary_linkdim(bl) == chi

    s = boundary_bond_ind(bl)
    @test hasind(bl, s) && hasind(bl, s')  # right = unprimed, left = primed

    # edges share the bond index of the bulk tensor
    v = randn(ComplexF64, chi)
    blL = close_boundary(bl, v; side=:left)
    blR = close_boundary(bl, v; side=:right)
    @test ndims(blL) == ndims(blR) == 2
    @test boundary_bond_ind(blL) == s
    @test boundary_bond_ind(blR) == s
    @test plev(only(uniqueinds(inds(blL), boundary_phys_ind(blL)))) == 0

    # product states
    @test is_product_boundary(to_boundary([1, 0]))
    @test n_boundary_sites(to_boundary([1, 0])) == 0
    @test boundary_linkdim(to_boundary([1, 0])) == 1

    # bad input is rejected with a helpful error
    @test_throws ErrorException check_boundary(random_itensor(ComplexF64, sp, il, ir))       # bonds not a prime pair
    @test_throws ErrorException check_boundary(random_itensor(ComplexF64, sp, il, ir, sp'))  # rank 4

    # folding doubles the bond dimension and keeps the convention
    rho = fold_boundary(bl)
    @test dim(boundary_phys_ind(rho)) == 4
    @test boundary_linkdim(rho) == chi^2
    @test check_boundary(rho) === rho

    # folding order: bra index is the *slow* one, ie. kron(conj(v), v), matching the
    # combiner(ket, bra) used to fold the W tensors. Only visible for complex states.
    v2 = normalize(randn(ComplexF64, 2))
    fv = fold_boundary(ITensor(v2, Index(2, "Site")))
    @test Array(fv, inds(fv)[1]) ≈ kron(conj(v2), v2)
    @test !isapprox(kron(conj(v2), v2), kron(v2, conj(v2)))
end


@testset "conjugation of the top boundary (product states)" begin
    # Regression: fw_tMPS used to dag(tr) while fw_tMPO did not, so a *complex* final state
    # gave an inconsistent network. tMPO and tMPS must apply the same convention.

    L = 5
    Nt = 3
    dt = 0.1
    mp = IsingParams(1.0, 0.7, 0.0)

    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:L]
    U = build_Ut(ss, Murg(), mp; dt)
    ts = [Index(2, "Site,time,t=$i") for i in 1:Nt]

    v0 = normalize(randn(ComplexF64, 2))
    vf = normalize(randn(ComplexF64, 2))

    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=v0)
    b = FwtMPOBlocks(tp)

    psi_t = MPS([ITensor(v0, s) for s in ss])
    for _ in 1:Nt
        psi_t = apply(U, psi_t; alg="naive", truncate=false)
    end

    # network with <vf| taken as given (no conjugation)
    ref_noconj = scalar(prod([psi_t[j] * ITensor(vf, ss[j]) for j in 1:L]))
    # network closed with the bra <vf|
    ref_bra = scalar(prod([psi_t[j] * ITensor(conj(vf), ss[j]) for j in 1:L]))
    @test !isapprox(ref_noconj, ref_bra)   # complex state: the two really differ

    for (dagger_tr, ref) in ((false, ref_noconj), (true, ref_bra))
        mpo = fw_tMPO(b, ts; tr=vf, dagger_tr)
        ll = fw_tMPS(b, ts; LR=:left, tr=vf, dagger_tr)
        rr = fw_tMPS(b, ts; LR=:right, tr=vf, dagger_tr)
        @test isapprox(contract_columns(ll, mpo, rr, L), ref; rtol=1e-6)
    end

    # the default (dagger_tr=true) is the same as conjugating the input by hand and not daggering
    @test isapprox(contract_columns(
        fw_tMPS(b, ts; LR=:left, tr=conj(vf), dagger_tr=false),
        fw_tMPO(b, ts; tr=conj(vf), dagger_tr=false),
        fw_tMPS(b, ts; LR=:right, tr=conj(vf), dagger_tr=false), L), ref_bra; rtol=1e-6)
end


@testset "non-product boundaries: unfolded transverse contraction" begin

    L = 5
    Nt = 3
    dt = 0.1
    mp = IsingParams(1.0, 0.7, 0.0)

    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:L]
    U = build_Ut(ss, Murg(), mp; dt)
    ts = [Index(2, "Site,time,t=$i") for i in 1:Nt]

    chi = 2
    il = Index(chi, "il")
    ir = Index(chi, "ir")
    A = random_itensor(ComplexF64, ss[2], il, ir)
    vL = randn(ComplexF64, chi)
    vR = randn(ComplexF64, chi)

    psi_i = ti_mps(A, ss[2], il, ir, vL, vR, ss)

    bl = boundary_tensor(A; phys=ss[2], left=il, right=ir)
    blL = close_boundary(bl, vL; side=:left)
    blR = close_boundary(bl, vR; side=:right)

    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl)
    b = FwtMPOBlocks(tp)

    psi_t = psi_i
    for _ in 1:Nt
        psi_t = apply(U, psi_t; alg="naive", truncate=false)
    end

    @testset "product final state" begin
        vf = ComplexF64[1.0, 0.3 + 0.45im]
        # the builders close with the bra <vf| by default (dagger_tr=true)
        ref = scalar(prod([psi_t[j] * ITensor(conj(vf), ss[j]) for j in 1:L]))

        mpo = fw_tMPO(b, ts; bl, tr=vf)
        ll = fw_tMPS(b, ts; LR=:left, bl=blL, tr=vf)
        rr = fw_tMPS(b, ts; LR=:right, bl=blR, tr=vf)

        # one extra site at the bottom for the initial state
        @test length(mpo) == length(ll) == length(rr) == Nt + 1
        @test siteinds(ll) == siteinds(rr)
        @test first(siteinds(rr)) == boundary_bond_ind(bl)

        @test contract_columns(ll, mpo, rr, L) ≈ ref
    end

    @testset "non-product final state" begin
        jl = Index(chi, "jl")
        jr = Index(chi, "jr")
        B = random_itensor(ComplexF64, ss[2], jl, jr)
        wL = randn(ComplexF64, chi)
        wR = randn(ComplexF64, chi)

        psi_f = ti_mps(B, ss[2], jl, jr, wL, wR, ss)
        ref = overlap_mps_noconj(dag(psi_f), psi_t)   # <psi_f| psi_t>, bra convention

        tr = boundary_tensor(B; phys=ss[2], left=jl, right=jr)
        trL = close_boundary(tr, wL; side=:left)
        trR = close_boundary(tr, wR; side=:right)

        mpo = fw_tMPO(b, ts; bl, tr)
        ll = fw_tMPS(b, ts; LR=:left, bl=blL, tr=trL)
        rr = fw_tMPS(b, ts; LR=:right, bl=blR, tr=trR)

        # extra site at the bottom *and* at the top
        @test length(mpo) == length(ll) == length(rr) == Nt + 2
        @test last(siteinds(rr)) == boundary_bond_ind(tr)

        @test contract_columns(ll, mpo, rr, L) ≈ ref
    end

    @testset "tMPO_in convenience builder" begin
        vf = ComplexF64[1.0, 0.3 + 0.45im]
        ref = scalar(prod([psi_t[j] * ITensor(conj(vf), ss[j]) for j in 1:L]))

        # from the raw MPS column, giving the legs explicitly
        mpo = tMPO_in(b, ts; init_tensor=A, init_physidx=ss[2], left=il, right=ir, tr=vf)
        @test length(mpo) == Nt + 1

        # the extra site index is the one of the boundary tensor built internally
        s_bdry = noprime(siteind(mpo, 1))
        blL2 = boundary_tensor(A * ITensor(vL, il); phys=ss[2], right=ir, bond_ind=s_bdry)
        blR2 = boundary_tensor(A * ITensor(vR, ir); phys=ss[2], left=il, bond_ind=s_bdry)

        ll = fw_tMPS(b, ts; LR=:left, bl=blL2, tr=vf)
        rr = fw_tMPS(b, ts; LR=:right, bl=blR2, tr=vf)

        @test contract_columns(ll, mpo, rr, L) ≈ ref

        # passing an already canonical boundary tensor keeps its bond index
        mpo2 = tMPO_in(b, ts; init_tensor=bl, init_physidx=boundary_phys_ind(bl), tr=vf)
        @test noprime(siteind(mpo2, 1)) == boundary_bond_ind(bl)
    end

    @testset "bond dimension 1 reduces to the product case" begin
        v0 = ComplexF64[0.6, 0.8 + 0.2im]
        vf = ComplexF64[1.0, 0.3 + 0.45im]

        i1 = Index(1, "i1")
        i2 = Index(1, "i2")
        A1 = ITensor(reshape(v0, 2, 1, 1), ss[2], i1, i2)
        bl1 = boundary_tensor(A1; phys=ss[2], left=i1, right=i2)
        e1 = ComplexF64[1]

        tp1 = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl1)
        b1 = FwtMPOBlocks(tp1)

        mpo_np = fw_tMPO(b1, ts; bl=bl1, tr=vf)
        ll_np = fw_tMPS(b1, ts; LR=:left, bl=close_boundary(bl1, e1; side=:left), tr=vf)
        rr_np = fw_tMPS(b1, ts; LR=:right, bl=close_boundary(bl1, e1; side=:right), tr=vf)

        tp0 = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=v0)
        b0 = FwtMPOBlocks(tp0)
        mpo_p = fw_tMPO(b0, ts; tr=vf)
        ll_p = fw_tMPS(b0, ts; LR=:left, tr=vf)
        rr_p = fw_tMPS(b0, ts; LR=:right, tr=vf)

        @test length(mpo_np) == length(mpo_p) + 1
        @test contract_columns(ll_np, mpo_np, rr_np, L) ≈ contract_columns(ll_p, mpo_p, rr_p, L)
    end
end


@testset "non-product boundaries: folded == |unfolded|^2" begin

    L = 5
    Nt = 3
    dt = 0.1
    mp = IsingParams(1.0, 0.7, 0.0)

    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:L]
    ts = [Index(2, "Site,time,t=$i") for i in 1:Nt]
    tsf = [Index(4, "Site,time_fold,t=$i") for i in 1:Nt]

    chi = 2
    il = Index(chi, "il")
    ir = Index(chi, "ir")
    A = random_itensor(ComplexF64, ss[2], il, ir)
    vL = randn(ComplexF64, chi)
    vR = randn(ComplexF64, chi)
    vf = ComplexF64[1.0, 0.3 + 0.45im]

    bl = boundary_tensor(A; phys=ss[2], left=il, right=ir)
    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl)

    b = FwtMPOBlocks(tp)
    amp = contract_columns(
        fw_tMPS(b, ts; LR=:left, bl=close_boundary(bl, vL; side=:left), tr=vf),
        fw_tMPO(b, ts; bl, tr=vf),
        fw_tMPS(b, ts; LR=:right, bl=close_boundary(bl, vR; side=:right), tr=vf),
        L)

    bf = FoldtMPOBlocks(tp)
    # the folded blocks fold the *bulk* initial state; close it with folded edge vectors
    rho0L = close_boundary(bf.rho0, fold_boundary(ITensor(vL, il)); side=:left)
    rho0R = close_boundary(bf.rho0, fold_boundary(ITensor(vR, ir)); side=:right)
    # the unfolded amplitude closes with the bra <vf| (dagger_tr=true), so the matching
    # folded projector is the fold of the closing vector conj(vf)
    Pf = fold_boundary(ITensor(conj(vf), ss[1]))

    @test boundary_linkdim(bf.rho0) == chi^2

    mpo_f = folded_tMPO(bf, tsf; fold_op=Pf)
    ll_f = folded_tMPS(bf, tsf; LR=:left, rho0=rho0L, fold_op=Pf)
    rr_f = folded_tMPS(bf, tsf; LR=:right, rho0=rho0R, fold_op=Pf)

    @test length(mpo_f) == length(ll_f) == length(rr_f) == Nt + 1

    @test contract_columns(ll_f, mpo_f, rr_f, L) ≈ abs2(amp)

    # the expval helpers rebuild the tMPO from siteinds(rr): they must skip the extra site
    @testset "expvals with a non-product initial state" begin
        rrn = rr_f
        lln = ll_f
        for _ in 1:2
            rrn = apply_column(mpo_f, rrn)
            lln = apply_column(mpo_f, lln)
        end

        idv = ITransverse.vectorized_identity(Index(4))
        norm_LR = expval_LR(lln, rrn, ITransverse.itensor_to_vector(idv), bf)
        @test norm_LR ≈ overlap_noconj(lln, apply_column(folded_tMPO(bf, tsf), rrn))

        evs = compute_expvals(lln, rrn, ["Z", "X"], bf)
        @test all(isfinite, values(evs))
        @test abs(evs["Z"]) <= 1 + 1e-8
    end
end


@testset "non-product boundaries: fwback" begin

    L = 5
    nfw = 2
    dt = 0.1
    mp = IsingParams(1.0, 0.7, 0.0)

    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:L]
    U = build_Ut(ss, Murg(), mp; dt)
    ts = [Index(2, "Site,time,t=$i") for i in 1:(2*nfw)]

    chi = 2
    il = Index(chi, "il")
    ir = Index(chi, "ir")
    A = random_itensor(ComplexF64, ss[2], il, ir)
    vL = randn(ComplexF64, chi)
    vR = randn(ComplexF64, chi)

    psi_i = ti_mps(A, ss[2], il, ir, vL, vR, ss)
    psi_t = psi_i
    for _ in 1:nfw
        psi_t = apply(U, psi_t; alg="naive", truncate=false)
    end
    ref = inner(psi_t, psi_t)   # <psi(t)|1|psi(t)>

    bl = boundary_tensor(A; phys=ss[2], left=il, right=ir)
    blL = close_boundary(bl, vL; side=:left)
    blR = close_boundary(bl, vR; side=:right)

    # the top boundary closes the backward branch, ie. it is the bra <psi_0|: same tensor as
    # the bottom one (but its own boundary bond index), conjugated by the default dagger_tr
    trc = boundary_tensor(A; phys=ss[2], left=il, right=ir)
    trL = close_boundary(trc, vL; side=:left)
    trR = close_boundary(trc, vR; side=:right)

    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl)
    b = FwtMPOBlocks(tp)

    mpo = fwback_tMPO(b, ts, 0, nfw, nfw, 0; bl, tr=trc, mid_op=[1, 0, 0, 1])
    ll = fwback_tMPS(b, ts; LR=:left, bl=blL, tr=trL)
    rr = fwback_tMPS(b, ts; LR=:right, bl=blR, tr=trR)

    @test length(mpo) == length(ll) == length(rr) == 2*nfw + 2

    @test contract_columns(ll, mpo, rr, L) ≈ ref
end


@testset "non-product boundaries: truncation and power method" begin

    Nt = 4
    dt = 0.1
    mp = IsingParams(1.0, 0.7, 0.0)

    ss = [addtags(sim(mp.phys_site), "Site") for _ in 1:3]
    ts = [Index(2, "Site,time,t=$i") for i in 1:Nt]

    chi = 2
    il = Index(chi, "il")
    ir = Index(chi, "ir")
    A = random_itensor(ComplexF64, ss[2], il, ir)
    A /= norm(A)
    vL = randn(ComplexF64, chi)
    vR = randn(ComplexF64, chi)
    vf = ComplexF64[1.0, 0.3 + 0.45im]

    bl = boundary_tensor(A; phys=ss[2], left=il, right=ir)
    tp = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=bl)
    b = FwtMPOBlocks(tp)

    mpo = fw_tMPO(b, ts; bl, tr=vf)
    ll = fw_tMPS(b, ts; LR=:left, bl=close_boundary(bl, vL; side=:left), tr=vf)
    rr = fw_tMPS(b, ts; LR=:right, bl=close_boundary(bl, vR; side=:right), tr=vf)

    # untruncated reference for one column application
    exact = overlap_noconj(ll, apply_column(mpo, rr))

    # generic MPS machinery must cope with the extra (differently sized) site
    @test length(gensym_renyi_entropies(rr)) == length(rr)
    @test !isnothing(orthogonalize(rr, 1))

    for alg in ("naive", "densitymatrix")
        rr_t, _ = tapply(mpo, rr; alg, cutoff=1e-14, maxdim=64)
        @test length(rr_t) == length(rr)
        @test overlap_noconj(ll, rr_t) ≈ exact
    end

    ll_t, rr_t, _ = tlrapply(ll, mpo, mpo, rr; alg="naiveRTM", cutoff=1e-14, maxdim=64)
    @test length(ll_t) == length(rr_t) == length(rr)
    @test isapprox(overlap_noconj(ll_t, rr_t),
                   overlap_noconj(apply_column(mpo, ll), apply_column(mpo, rr)); rtol=1e-6)

    # the power method keeps the extra boundary site around
    pm_params = PMParams(; truncp=(; cutoff=1e-12, maxdim=32, alg="naive"),
        itermax=20, eps_converged=1e-10, opt_method=:sym, normalization="norm")
    psi, _ = powermethod_sym(rr, mpo, pm_params)
    @test length(psi) == Nt + 1
    @test siteinds(psi) == siteinds(rr)
end
