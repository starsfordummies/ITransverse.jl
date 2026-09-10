using ITensors, ITensorMPS, ITransverse
using Observers

# ── parameters ───────────────────────────────────────────────────────────────

LL   = 20        # chain length
Nt   = 50        # number of TEBD steps
dt   = 0.1       # time step
g    = 0.5       # transverse field
h    = 0.0       # longitudinal field
J    = 1.0       # Ising coupling

maxdim = 128
cutoff = 1e-12

# ── model setup ───────────────────────────────────────────────────────────────

mp = IsingParams(J, g, h)
tp = tMPOParams(mp; dt, init_state=up_state)

# ── observer: ⟨Z⟩ at midchain + bond dimension ───────────────────────────────

obs = observer(
    "Z"    => (; state) -> expect(state, "Z")[halfsite(state)],
    "chi"  => (; state) -> maxlinkdim(state),
    "time" => (; time)  -> time,
)

# ── time evolution ────────────────────────────────────────────────────────────

psi_t = tebd(LL, tp, Nt; maxdim, cutoff, (observer!)=obs)

# ── results ───────────────────────────────────────────────────────────────────

ts   = obs[!, "time"]
Z_ev = real.(obs[!, "Z"])
chis = obs[!, "chi"]

@show Z_ev[end]
@show chis[end]

# To plot (requires Plots loaded):
# using Plots
# plot(ts, Z_ev; xlabel="t", ylabel="⟨Z⟩", label="midchain", marker=:o)
# plot(ts, chis; xlabel="t", ylabel="χ", label="bond dim")
