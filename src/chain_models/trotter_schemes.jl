"""
Abstract supertype for Trotter-decomposition recipes used to build
local time-evolution tensors ``\\exp(-i H_{\\rm local}\\,\\delta t)``.

Concrete subtypes:
- [`Murg`](@ref)    – Murg/Vidal exact two-site decomposition
- [`SymSVD`](@ref)  – symmetric SVD Trotter decomposition
- [`Floquet`](@ref) – stroboscopic Floquet gates

Dispatched by [`expH`](@ref) together with the [`ModelParams`](@ref) subtype.
"""
abstract type ExpHRecipe end

""" Murg/Vidal-style exact decomposition of the two-site gate ``\\exp(-i H_{12}\\,\\delta t)``
into a product of MPO tensors. Supported for [`IsingParams`](@ref) and [`PottsParams`](@ref). """
struct Murg    <: ExpHRecipe end

""" Symmetric SVD Trotter decomposition: first-order split
``\\exp(-i H \\delta t) \\approx \\prod_k \\exp(-i h_k \\delta t)``
assembled via SVD. Supported for [`IsingParams`](@ref), [`PottsParams`](@ref), and [`XXZParams`](@ref). """
struct SymSVD  <: ExpHRecipe end

""" Floquet (stroboscopic) gates: ``\\exp(-i J XX)\\exp(-i \\lambda X)\\exp(-i g Z)``.
Only valid for [`IsingParams`](@ref). """
struct Floquet <: ExpHRecipe end

"""
    expH(sites, mp::ModelParams, scheme::ExpHRecipe; dt) -> MPO

Build the local time-evolution MPO ``U = \\exp(-i H \\delta t)`` (or the imaginary-time
equivalent) on `sites` using the coupling constants in `mp` and the decomposition
recipe `scheme`.

| `mp` type     | supported `scheme`        |
|:-------------|:--------------------------|
| `IsingParams` | `Murg()`, `SymSVD()`, `Floquet()` |
| `PottsParams` | `Murg()`, `SymSVD()`      |
| `XXZParams`   | `SymSVD()`                |

See [Algorithms](@ref) for a description of each scheme.
"""
expH(sites, mp::IsingParams, ::Murg;    dt) = expH_ising_murg(sites, mp; dt)
expH(sites, mp::IsingParams, ::SymSVD;  dt) = expH_ising_symm_svd(sites, mp; dt)
expH(sites, mp::IsingParams, ::Floquet; dt) = expH_ising_floquet(sites, mp; dt)

# ── Potts ────────────────────────────────────────────────────────────────────
expH(sites, mp::PottsParams, ::Murg;   dt) = expH_potts_murg(sites, mp; dt)
expH(sites, mp::PottsParams, ::SymSVD; dt) = expH_potts_symmetric_svd(sites, mp; dt)

# ── XXZ ──────────────────────────────────────────────────────────────────────
expH(sites, mp::XXZParams, ::SymSVD; dt) = expH_XXZ_svd(sites, mp; dt)

# ── Per-model defaults (used by tMPOParams(mp::ModelParams; ...)) ────────────
default_scheme(::IsingParams) = Murg()
default_scheme(::PottsParams) = Murg()
default_scheme(::XXZParams)   = SymSVD()
default_scheme(::NoParams)    = Murg()

default_bl(::IsingParams) = [1, 0]
default_bl(::PottsParams) = [1, 0, 0]
default_bl(::XXZParams)   = [1, 0]
default_bl(::NoParams)    = [1, 0]
