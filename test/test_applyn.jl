using ITensors, ITensorMPS
using Test

using ITransverse
using ITransverse.ITenUtils: check_mps_sanity

b = FoldtMPOBlocks(ising_tp())
ss = siteinds(4, 10)


@testset "applyn/contractn" begin
    psi = random_mps(ss, linkdims=7)
    oo = random_mpo(ss) + random_mpo(ss)
    oo2 = random_mpo(ss) + random_mpo(ss)

    opsi1 = applyn(oo, psi)
    opsi2 = apply(oo, psi, alg="naive")

    @test opsi1 ≈ opsi2

    oo3 = applyn(oo, oo2)
    oo4 = apply(oo,oo2; alg="naive", truncate=false)

    @test oo3 ≈ oo4
end


@testset "Truncations" begin
    oo = random_mpo(ss) + random_mpo(ss)
    oo2 = random_mpo(ss) + random_mpo(ss)

    psi = random_mps(ss, linkdims=7)

    for jj = 1:10
        psi = applyn(oo, psi; maxdim = 12)
    end

    @test maxlinkdim(psi) <= 12

    psi = random_mps(ss, linkdims=2)

    for jj = 1:6
        psi = applyn(oo, psi; cutoff=1e-6)
    end

    @test maxlinkdim(psi) < 2^7


    tp = TruncParams(1e-20, 24)

    psi = random_mps(ss, linkdims=2)

    for jj = 1:7
        psi = applyn(oo, psi; truncp=tp)
    end

    @test maxlinkdim(psi) <= 24

    # Now applyns()

    psi = random_mps(ss, linkdims=7)

    for jj = 1:10
        psi = applyns(oo, psi; maxdim = 12)
    end

    @test maxlinkdim(psi) <= 12

    psi = random_mps(ss, linkdims=2)

    for jj = 1:6
        psi = applyns(oo, psi; cutoff=1e-6)
    end

    @test maxlinkdim(psi) < 2^7


    tp = TruncParams(1e-20, 24)

    psi = random_mps(ss, linkdims=2)

    for jj = 1:7
        psi = applyns(oo, psi; truncp=tp)
    end

    @test maxlinkdim(psi) <= 24


end

@testset "various apply" begin


b = FoldtMPOBlocks(ising_tp())
ss = siteinds(4, 10)

n_ext = 3 

oooR = ITransverse.folded_tMPO_ext(b, ss; LR=:right, n_ext, fold_op=[1,0,0,-1])
@test count(isempty, siteinds(oooR,plev=0)) == n_ext
@test count(isempty, siteinds(oooR,plev=1)) == 0

oooL = ITransverse.folded_tMPO_ext(b, ss; LR=:left, n_ext, fold_op=[1,0,0,-1])
@test count(isempty, siteinds(oooL,plev=1)) == n_ext
@test count(isempty, siteinds(oooL,plev=0)) == 0


# Apply Right MPO to (shorter) Right tMPS: Extend 
psiR = random_mps(ss[1:end-n_ext], linkdims=20)

psiR_ext = applyn(oooR,psiR)
@test siteinds(psiR_ext) == ss
@test check_mps_sanity(psiR_ext)


psiR_ext = applyn(oooR,psiR, truncate=true)
@test siteinds(psiR_ext) == ss
@test check_mps_sanity(psiR_ext)


psiR_ext = applyn(oooR,psiR, cutoff=1e-12)
@test siteinds(psiR_ext) == ss
@test check_mps_sanity(psiR_ext)

# Apply Left tMPO to (shorter) Left tMPS: Extend
psiL = random_mps(ss[1:end-n_ext], linkdims=20)
psiL_ext = applyns(oooL,psiL)
@test check_mps_sanity(psiL_ext)

@test siteinds(psiL_ext) == ss


# Apply Right MPO to (equal length) Left tMPS: Chop 

psiL = random_mps(ss, linkdims=20)

psiL_chop = applyns(oooR,psiL)
@test check_mps_sanity(psiL_chop)
@test length(psiL_chop) == length(psiL)-n_ext

# Apply Left tMPO to (equal length) Right tMPS: Chop
psiR = random_mps(ss, linkdims=20)
psiR_chop = applyn(oooL,psiR)
@test check_mps_sanity(psiR_chop)
@test length(psiR_chop) == length(psiR)-n_ext


# What if they're only long 2 

ss = siteinds(4, 2)
psiR = random_mps(ss, linkdims=20)
psiR.llim
psiR.rlim
oooL = ITransverse.folded_tMPO_ext(b, ss; LR=:left, n_ext=1, fold_op=[1,0,0,1])

kk = applyn(oooL,psiR)


ITransverse.ITenUtils.contract_dangling!(kk)
kk.llim
kk.rlim
 orthogonalize(kk, length(kk))
 normalize(kk)


 #expvals
ss = siteinds(4, 10)

psiL = random_mps(ss, linkdims=20)
oooR = ITransverse.folded_tMPO_ext(b, ss; LR=:right, n_ext=3, fold_op=[1,0,0,-1])
psiR = random_mps(ss[1:end-3], linkdims=20)

expval_LR(psiL, oooR, psiR)


ss = siteinds("S=1/2", 10)
psi = random_mps(ss, linkdims=20)
oo = random_mpo(ss) + random_mpo(ss)
pp = random_mpo(ss) + random_mpo(ss)

opsi = apply(oo, psi; cutoff=1e-12)
opsi2, sv = tapply(oo, psi; cutoff=1e-12)
opsi3, sv = tapply(oo, psi; alg="densitymatrix", cutoff=1e-12)
opsi4, sv = tapply(oo, psi; alg="zipup", cutoff=1e-12)

@test opsi ≈ opsi2
@test opsi ≈ opsi3
@test opsi ≈ opsi4

end