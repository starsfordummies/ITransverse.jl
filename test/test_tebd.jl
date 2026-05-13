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

# ── ⟨Z⟩ via observer ────────────────────────────────────────────────────────
@testset "collect Z via observer" begin
    obs_tp  = observer("Z" => (; state) -> expect(state, "Z")[halfsite(state)])
    obs_mpo = observer("Z" => (; state) -> expect(state, "Z")[halfsite(state)])
    tebd(N_TEBD,  tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64, (observer!)=obs_tp)
    tebd(psi0_up, Ut_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64, (observer!)=obs_mpo)
    evs_z_tp  = ComplexF64.(obs_tp[!,  "Z"])
    evs_z_mpo = ComplexF64.(obs_mpo[!, "Z"])

    @test length(evs_z_tp)  == Nt_TEBD
    @test eltype(evs_z_tp)  == ComplexF64
    @test all(abs.(evs_z_tp) .<= 1.0 + 1e-8)

    # tp and explicit-MPO variants agree
    @test evs_z_tp ≈ evs_z_mpo atol=1e-6

    # compare against BenchData: ⟨Z⟩, up_state, g=0.4
    @test real.(evs_z_tp) ≈ bench_Z_04_up[1:Nt_TEBD] atol=BENCH_TOL
end

# ── multi-operator observer ──────────────────────────────────────────────────
@testset "multi-operator observer" begin
    obs = observer(
        "Z"   => (; state) -> expect(state, "Z")[halfsite(state)],
        "X"   => (; state) -> expect(state, "X")[halfsite(state)],
        "chi" => (; state) -> maxlinkdim(state),
    )
    psi_t = tebd(N_TEBD, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64,
                 (observer!)=obs)

    @test all(haskey(obs.data, k) for k in ["Z", "X", "chi"])
    @test length(obs[!, "Z"]) == length(obs[!, "X"]) == length(obs[!, "chi"]) == Nt_TEBD
    @test all(obs[!, "chi"] .>= 1)
    @test psi_t isa MPS && length(psi_t) == N_TEBD
    @test norm(psi_t) ≈ 1.0 atol=1e-8

    # MPS-init variant gives the same Z track
    obs2 = observer(
        "Z" => (; state) -> expect(state, "Z")[halfsite(state)],
        "X" => (; state) -> expect(state, "X")[halfsite(state)],
    )
    tebd(psi0_up, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=64,
         (observer!)=obs2)
    @test obs[!, "Z"] ≈ obs2[!, "Z"] atol=1e-6
end

@testset "observer vs BenchData (g=0.4, up_state)" begin
    obs = observer(
        "Z" => (; state) -> expect(state, "Z")[halfsite(state)],
        "X" => (; state) -> expect(state, "X")[halfsite(state)],
    )
    tebd(N_TEBD, tp_up, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=128,
         (observer!)=obs)

    @test real.(obs[!, "Z"]) ≈ bench_Z_04_up[1:Nt_TEBD] atol=BENCH_TOL
    # bench_X_04_up is all zeros (symmetry: ⟨X⟩=0 for up state, h=0)
    @test real.(obs[!, "X"]) ≈ bench_X_04_up[1:Nt_TEBD] atol=BENCH_TOL
end

@testset "observer vs BenchData (g=0.4, plus_state)" begin
    obs = observer(
        "Z" => (; state) -> expect(state, "Z")[halfsite(state)],
        "X" => (; state) -> expect(state, "X")[halfsite(state)],
    )
    tebd(N_TEBD, tp_plus, Nt_TEBD; normalize=true, cutoff=1e-12, maxdim=128,
         (observer!)=obs)

    @test real.(obs[!, "Z"]) ≈ bench_Z_04_plus[1:Nt_TEBD] atol=BENCH_TOL
    @test real.(obs[!, "X"]) ≈ bench_X_04_plus[1:Nt_TEBD] atol=BENCH_TOL
end
