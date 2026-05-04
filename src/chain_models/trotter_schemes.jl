abstract type ExpHRecipe end

struct Murg    <: ExpHRecipe end
struct SymSVD  <: ExpHRecipe end
struct Floquet <: ExpHRecipe end

# ── Ising ────────────────────────────────────────────────────────────────────
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
