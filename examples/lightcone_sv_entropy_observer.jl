using ITensors, ITensorMPS
using JLD2
using ITransverse
using Plots; gr(size=(900, 500))

# ── Helper: SVD entropy from truncation singular values ─────────────
"""
    svd_entropy_from_sv(sv::AbstractMatrix) -> Vector{Float64}

Compute Von Neumann ("SVD") entropy at every cut from the TruncLR.sv matrix
returned by tlrapply.  Each row `sv[k, :]` holds the SVD singular values at
cut k.  Treat σ² as a probability distribution and compute `S = -Σ p log p`.

This is the Shannon entropy of the singular-value weight distribution per cut —
not the RTM eigenvalue spectrum.
"""
function svd_entropy_from_sv(sv::AbstractMatrix)
    ncuts = size(sv, 1)
    S = Vector{Float64}(undef, ncuts)
    for k in 1:ncuts
        σ² = real(sv[k, :] .^ 2)
        s2sum = sum(σ²)
        if s2sum < 1e-30
            S[k] = 0.0
        else
            p = σ² ./ s2sum
            S[k] = -sum(p .* log.(max.(p, 1e-30)))
        end
    end
    return S
end

# ── Main entry point ────────────────────────────────────────────────
function run_sv_entropy_cone(Ngrow::Int)
    JXX, hz, gx = 1.0, 1.05, 0.5
    dt          = 0.1
    cutoff      = 1e-12
    maxdim      = 64

    mp     = IsingParams(JXX, hz, gx)
    tp     = tMPOParams(mp; dt, scheme=Murg(), nbeta=0, init_state=up_state)
    b      = FoldtMPOBlocks(tp)
    c0     = init_cone(b, 3)

    truncp  = (; cutoff, maxdim, direction=:left, alg="RTM")
    cone_params = ConeParams(; truncp, opt_method=:sym, optimize_op=[1, 0, 0, 1])

    # Observers using the checkpoint system (state tuple now has sv field)
    f_obs = (
        SVN     = s -> vn_entanglement_entropy(s.R),
        S_SVD   = s -> generalized_svd_vn_entropy(s.L, s.R),
        overlap = s -> overlap_noconj(s.L, s.R),
        SvdEnt  = s -> svd_entropy_from_sv(Array(s.sv)),
        chi     = s -> maxlinkdim(s.L),
        expvals = s -> compute_expvals(s.L, s.R, ["Z","X"], s.b)

    )

    f_state = (
        L  = s -> s.L,
        R  = s -> s.R,
        b  = s -> s.b,
        sv = s -> s.sv
    )

    cp = DoCheckpoint(
        "SvdEnt_cone_ising_Nt$(Ngrow).jld2";
        params=Dict("tparams" => tp, "cparams" => cone_params),
        save_at   = [Ngrow],
        f_obs     = f_obs,
        f_savestate = f_state,
    )

    # Run the light cone using existing engine
    psiL, psiR, checkpt = run_cone(c0, b, cone_params, cp, Ngrow)

    # Extract results from checkpoint history
    obs_hist  = checkpt.obs_hist
    steps     = checkpt.steps
    ov_hist   = obs_hist[:overlap]
    SvdEnt_history = obs_hist[:SvdEnt]
    SVN_R       = obs_hist[:SVN]
    chi_hist    = obs_hist[:chi]
    evs = obs_hist[:expvals]

    ncuts_final = length(SvdEnt_history[end])

    println("\n═══ Mid-bond observables (all steps) ══════════════════════════════")
    println("step   t        χ        overlap ⟨L|R>     SvdEnt(mid)")
    for i in eachindex(steps)
        t = steps[i] * dt / Ngrow
        ncuts = length(SvdEnt_history[i])
        mid = div(ncuts, 2)
        println("$(steps[i]): $(t)   $(chi_hist[i])   $(ov_hist[i])       $(SvdEnt_history[i][mid])")
    end

    # Per-cut SVD entropy at final step
    println("\n═══ Per-cut SVD entropy at t=$(steps[end]*dt/Ngrow) ═════════════════════===")
    last_SvdEnt = SvdEnt_history[end]
    for k in 1:ncuts_final
        println("  cut $k: S₁ = $(last_SvdEnt[k])")
    end

    # VN( |R> ) at mid cut for comparison
    SVN_R_mid_all = Vector{Float64}(undef, length(steps))
    for i in eachindex(steps)
        r_len = length(SVN_R[i])
        r_mid = div(r_len, 2)
        SVN_R_mid_all[i] = SVN_R[i][r_mid]
    end

    println("\n── Mid-bond comparison ═══════")
    println("      S₁_SVD(mid)    VN(RDM,mid)")
    for i in eachindex(steps)
        i_mid = div(length(SvdEnt_history[i]), 2)
        println("t=$(steps[i]*dt/Ngrow):  $(SvdEnt_history[i][i_mid])         $(SVN_R_mid_all[i])")
    end

    return steps, SvdEnt_history, SVN_R_mid_all, ncuts_final, chi_hist, checkpt, evs
end

# ── Run it ───────────────────────────────────────────────────────────
steps, SvdEnt_hist, SVN_R_mid, ncuts, chi_hist, checkpt, evs = run_sv_entropy_cone(40)

# ── Plot all cuts ──────────────────────────────────────────────────
# Cut count grows with the cone, so only plot cut k over the steps where it exists.
p_all = plot(label="S₁ SVD (all cuts)", ylabel="S₁ SVD", xlabel="step", 
             legend=:topleft, title="SVD Entropy — All Cuts")
for k in 1:ncuts
    idxs = findall(s -> length(s) >= k, SvdEnt_hist)
    plot!(steps[idxs], [SvdEnt_hist[i][k] for i in idxs], label="cut $k", linewidth=1.5)
end


# ── Save entropies and max bond dimension ──────────────────────────
jldsave("SvdEnt_data_Nt20.jld2";
    steps, SvdEnt_hist, SVN_R_mid, max_traj, chi_hist, chi_max=maximum(chi_hist), evs)
