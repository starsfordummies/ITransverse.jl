# Algorithms

ITransverse uses **two orthogonal dispatch mechanisms** to select algorithms at run time without conditional branches in the calling code.

---

## 1. `Algorithm` string dispatch (`ITensors.Algorithm`)

The `@Algorithm_str` macro from **ITensors.jl** turns a string literal into a singleton type, so that Julia's ordinary method dispatch can be used to select an implementation:

```julia
using ITensors: @Algorithm_str, Algorithm

tcontract(::Algorithm"naive", A, ψ; kwargs...)
tcontract(::Algorithm"densitymatrix", A, ψ; kwargs...)
tcontract(::Algorithm"zipup", A, ψ; kwargs...)
```

Every public entry point accepts an `alg` keyword that is converted transparently:

```julia
tapply(A, ψ; alg="densitymatrix", maxdim=128, cutoff=1e-12)
# dispatches to tcontract(Algorithm("densitymatrix"), A, ψ; ...)
```

### Available `tcontract` / `tapply` algorithms

| Algorithm string | Description |
|---|---|
| `"naive"` | Direct site-by-site MPO-MPS contraction followed by a single right-to-left SVD sweep ([`ttruncate!`](@ref)). Allows extension (MPO longer than MPS). |
| `"densitymatrix"` | Density-matrix renormalisation step: builds the reduced density matrix at each bond and diagonalises it to find the optimal truncation basis. Analogous to the `densitymatrix` algorithm in ITensors.jl. |
| `"zipup"` | Sequential QR / factorisation sweep from left to right, contracting one site at a time. |
| `"naiveRTMsym"` | Symmetric variant of `"naive"`: keeps the MPS in symmetric (right-canonical) form after contraction. |
| `"naiveRTMsymRTM"` | As above but the output is also cast to an RTM (reduced transfer matrix) form. |
| `"RTMsym"` | Density-matrix algorithm applied to a symmetric MPS (RTM). |

The algorithm string is stored in the `alg` field of the named-tuple `truncp` that is threaded through [`PMParams`](@ref), [`ConeParams`](@ref), and the truncation-sweep helpers, so the same keyword applies everywhere:

```julia
pm = PMParams(; truncp=(cutoff=1e-12, maxdim=256, alg="densitymatrix", direction=:right))
cone = ConeParams(; truncp=pm.truncp)
```

---

## 2. `ExpHRecipe` dispatch

`ExpHRecipe` is an **abstract type** whose concrete subtypes represent distinct recipes for building the local time-evolution tensors $\exp(-i H_{\rm local} \, \delta t)$.  The main entry point is [`expH`](@ref), which dispatches on both the model parameter type and the recipe:

```julia
abstract type ExpHRecipe end

struct Murg    <: ExpHRecipe end   # Murg/Vidal-style exact two-site decomposition
struct SymSVD  <: ExpHRecipe end   # symmetric SVD Trotter decomposition
struct Floquet <: ExpHRecipe end   # Floquet (stroboscopic) gates
```

The dispatch table looks like:

```julia
expH(sites, mp::IsingParams, ::Murg;    dt)  # → expH_ising_murg
expH(sites, mp::IsingParams, ::SymSVD;  dt)  # → expH_ising_symm_svd
expH(sites, mp::IsingParams, ::Floquet; dt)  # → expH_ising_floquet

expH(sites, mp::PottsParams, ::Murg;    dt)  # → expH_potts_murg
expH(sites, mp::PottsParams, ::SymSVD;  dt)  # → expH_potts_symmetric_svd

expH(sites, mp::XXZParams,   ::SymSVD;  dt)  # → expH_XXZ_svd
```

The scheme is stored in [`tMPOParams`](@ref) and propagates automatically to all tMPO builders:

```julia
tp = tMPOParams(IsingParams(1.0, 0.4, 0.0); dt=0.1, scheme=SymSVD())
```

### Per-model defaults

Each `ModelParams` subtype provides a default `ExpHRecipe` through `default_scheme`:

| Model | Default scheme |
|---|---|
| `IsingParams` | `Murg()` |
| `PottsParams` | `Murg()` |
| `XXZParams`   | `SymSVD()` |
| `NoParams`    | `Murg()` |

---

## 3. How the two dispatch mechanisms compose

A typical call chain looks like this:

```
tMPOParams(mp; scheme=Murg())
    └─ FoldtMPOBlocks(tp)          # calls expH(sites, mp, Murg(); dt=tp.dt)
           └─ run_cone / powermethod_op
                  └─ tlrapply(...; alg="densitymatrix", ...)
                         └─ tcontract(Algorithm"densitymatrix", A, ψ; ...)
```

The `ExpHRecipe` controls **what network** is built; the `Algorithm` string controls **how** the resulting MPO-MPS product is compressed.

---

## API references

See the [Truncation & Sweeps](api/truncation.md) page for the full `tcontract`
documentation.

```@docs
ttruncate!
expH
ExpHRecipe
Murg
SymSVD
Floquet
```
