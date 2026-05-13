# TEBD

Time-evolving block decimation (TEBD) for real- and imaginary-time evolution.

Measurements are collected via an Observers.jl observer passed as `(observer!)=obs`.
Each observer function receives `state`, `step`, and `time` as keyword arguments.

## Example

```julia
using Observers: observer

obs = observer(
    "Z"   => (; state) -> expect(state, "Z")[halfsite(state)],
    "chi" => (; state) -> maxlinkdim(state),
)

tp = tMPOParams(IsingParams(1.0, 0.4, 0.0); dt=0.1)
psi_t = tebd(20, tp, 50; maxdim=128, cutoff=1e-12, (observer!)=obs)

obs[!, "Z"]    # Vector of ⟨Z⟩ values at each step
obs[!, "chi"]  # bond dimension history
```

## API

```@docs
tebd
```
