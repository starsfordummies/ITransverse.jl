using ITensors, ITensorMPS
using ITransverse
using ITransverse.BenchData
using Test

# ── shared setup ─────────────────────────────────────────────────────────────
# N large enough that the half-site is away from boundaries; Nt kept short for speed.
const N_TEBD  = 20
const Nt_TEBD = 10
const DT_TEBD = 0.1
const BENCH_TOL = 1e-2   # loose enough to tolerate Trotter + truncation errors

# integrable transverse-field Ising, spin-up initial state  (g=0.4)
tp_up   = tMPOParams(IsingParams(1.0, 0.4, 0.0); dt=DT_TEBD, init_state=up_state)
# same model, plus_state initial condition
tp_plus = tMPOParams(IsingParams(1.0, 0.4, 0.0); dt=DT_TEBD, init_state=plus_state)

ss   = siteinds("S=1/2", N_TEBD)
psi0_up   = pMPS(ss, up_state)
psi0_plus = pMPS(ss, plus_state)
Ut_up   = ITransverse.build_Ut(ss, tp_up)

# ── tebd(psi0, Ut, Nt) ────────────────────────────────────────────────────────
@testset "tebd with explicit MPO" begin
    psi_t = tebd(psi0_up, Ut_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)

    @test psi_t isa MPS
    @test length(psi_t) == N_TEBD
    @test norm(psi_t) ≈ 1.0 atol=1e-8
end

# ── direct observer! API ─────────────────────────────────────────────────────
@testset "direct observer! API" begin
    obs = observer(
        "Z"    => (; state) -> expect(state, "Z")[halfsite(state)],
        "chi"  => (; state) -> maxlinkdim(state),
        "time" => (; time)  -> time,
    )
    psi_t = tebd(psi0_up, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64,
                 (observer!)=obs)

    @test length(obs[!, "Z"])   == Nt_TEBD
    @test length(obs[!, "chi"]) == Nt_TEBD
    # time column should be dt*step
    @test obs[!, "time"] ≈ DT_TEBD .* (1:Nt_TEBD) atol=1e-12
    # Z track must match tebd_z
    evs_z = tebd_z(psi0_up, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)
    @test ComplexF64.(obs[!, "Z"]) ≈ evs_z atol=1e-6
end

# ── tebd(psi0, tp, Nt) and tebd(LL, tp, Nt) ──────────────────────────────────
@testset "tebd calling conventions agree" begin
    psi_a = tebd(psi0_up, Ut_up,  Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)
    psi_b = tebd(psi0_up, tp_up,  Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)
    psi_c = replace_siteinds(tebd(N_TEBD,  tp_up,  Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64), ss)

    @test inner(psi_a, psi_b) ≈ 1.0 atol=1e-6
    @test inner(psi_a, psi_c) ≈ 1.0 atol=1e-6
    @test length(psi_c) == N_TEBD
    @test norm(psi_c) ≈ 1.0 atol=1e-8
end

# ── tebd_z ────────────────────────────────────────────────────────────────────
@testset "tebd_z" begin
    evs_z_tp  = tebd_z(N_TEBD, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)
    evs_z_mpo = tebd_z(psi0_up, Ut_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)

    @test length(evs_z_tp)  == Nt_TEBD
    @test eltype(evs_z_tp)  == ComplexF64
    @test all(abs.(evs_z_tp) .<= 1.0 + 1e-8)

    # tp and explicit-MPO variants agree
    @test evs_z_tp ≈ evs_z_mpo atol=1e-6

    # compare against BenchData: ⟨Z⟩, up_state, g=0.4
    @test real.(evs_z_tp) ≈ bench_Z_04_up[1:Nt_TEBD] atol=BENCH_TOL
end

# ── tebd_ev ───────────────────────────────────────────────────────────────────
@testset "tebd_ev structure" begin
    ops = ["Z", "X"]
    evs, psi_t = tebd_ev(N_TEBD, tp_up, Nt_TEBD, ops; normalize=true, cutoff=1e-12, maxdim=64)

    @test haskey(evs, "Z") && haskey(evs, "X") && haskey(evs, "chis")
    @test length(evs["Z"]) == length(evs["X"]) == length(evs["chis"]) == Nt_TEBD
    @test all(evs["chis"] .>= 1)
    @test psi_t isa MPS && length(psi_t) == N_TEBD
    @test norm(psi_t) ≈ 1.0 atol=1e-8

    # "Z" track must agree with tebd_z
    evs_z = tebd_z(N_TEBD, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64)
    @test evs["Z"] ≈ evs_z atol=1e-6

    # MPS-init variant must give the same result
    evs2, _ = tebd_ev(psi0_up, tp_up, Nt_TEBD, ops; normalize=true, cutoff=1e-12, maxdim=64)
    @test evs["Z"] ≈ evs2["Z"] atol=1e-6
end

@testset "tebd_ev vs BenchData (g=0.4, up_state)" begin
    ops = ["Z", "X"]
    evs, _ = tebd_ev(N_TEBD, tp_up, Nt_TEBD, ops; normalize=true, cutoff=1e-12, maxdim=128)

    @test real.(evs["Z"]) ≈ bench_Z_04_up[1:Nt_TEBD] atol=BENCH_TOL
    # bench_X_04_up is all zeros (symmetry: ⟨X⟩=0 for up state, h=0)
    @test real.(evs["X"]) ≈ bench_X_04_up[1:Nt_TEBD] atol=BENCH_TOL
end

@testset "tebd_ev vs BenchData (g=0.4, plus_state)" begin
    ops = ["Z", "X"]
    evs, _ = tebd_ev(N_TEBD, tp_plus, Nt_TEBD, ops; normalize=true, cutoff=1e-12, maxdim=128)

    @test real.(evs["Z"]) ≈ bench_Z_04_plus[1:Nt_TEBD] atol=BENCH_TOL
    @test real.(evs["X"]) ≈ bench_X_04_plus[1:Nt_TEBD] atol=BENCH_TOL
end
