# Chain Models

Types and functions for defining spin-chain Hamiltonians and their time-evolution gates.

## Parameter types

```@docs
ModelParams
NoParams
IsingParams
PottsParams
XXZParams
```

## Hamiltonian builders

- `H_ising` – Ising Hamiltonian MPO: ``H = -J_{\rm two}\sum XX - g_\perp\sum Z - h_\parallel\sum X``
- `H_potts` / `H_potts_manual` – 3-state Potts Hamiltonian MPO
- `build_H` – dispatch wrapper that calls the right builder from a `ModelParams`

```@docs
H_ising
```

## Time-evolution gate builders

See [`expH`](@ref) (documented on the [Algorithms](../algorithms.md) page).

`build_Ut(sites, scheme, mp; dt)` – thin wrapper that calls `expH` and assembles the full-chain MPO.

## Trotter schemes

See the [Algorithms](../algorithms.md) page for a full description.
[`ExpHRecipe`](@ref), [`Murg`](@ref), [`SymSVD`](@ref), [`Floquet`](@ref).

## tMPO parameter type

```@docs
tMPOParams
ising_tp
```
