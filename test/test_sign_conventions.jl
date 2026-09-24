# All U(dt) builders must implement exp(-i*H*dt) for the matching H builder (H = -(...) convention).
using Test, ITensors, ITensorMPS, LinearAlgebra
using ITransverse
using ITransverse: IsingParams, PottsParams, XXZParams, expH, Murg, SymSVD, Floquet

_dense(M::MPO, ss) = (T = reduce(*, M); n = prod(dim.(ss)); reshape(array(T, [prime.(ss); ss]...), n, n))

@testset "sign conventions: U(dt) = exp(-iHdt)" begin
    N, dt = 4, 0.05
    ss  = siteinds("S=1/2", N)
    sp  = siteinds("S=1", N)
    chk(U, H, tol) = norm(U - exp(-im * dt * H)) < tol && norm(U - exp(im * dt * H)) > 10 * tol

    mp = IsingParams(1.0, 0.7, 0.3)
    H  = _dense(ITransverse.H_ising(ss, mp), ss)
    @test chk(_dense(expH(ss, mp, Murg();    dt), ss), H, 2e-3)
    @test chk(_dense(expH(ss, mp, SymSVD();  dt), ss), H, 2e-3)
    @test chk(_dense(expH(ss, mp, Floquet(); dt), ss), H, 5e-2)   # first order

    xp = XXZParams(1.0, 0.6, 0.4)
    Hx = _dense(ITransverse.H_XXZ(ss, xp), ss)
    @test chk(_dense(expH(ss, xp, SymSVD(); dt), ss), Hx, 5e-3)
    @test norm(Hx - _dense(ITransverse.H_XXZ_SpSm(ss, 1.0, 0.6, 0.4), ss)) < 1e-12
    xp0 = XXZParams(1.0, 0.0, 0.0)
    @test chk(_dense(ITransverse.expH_XX_svd(ss, xp0; dt), ss), _dense(ITransverse.H_XXZ(ss, xp0), ss), 5e-3)

    pp = PottsParams(1.0, 0.5)
    Hp = _dense(ITransverse.H_potts(sp, pp), sp)
    @test chk(_dense(expH(sp, pp, Murg();   dt), sp), Hp, 5e-3)
    @test chk(_dense(expH(sp, pp, SymSVD(); dt), sp), Hp, 5e-3)
end
