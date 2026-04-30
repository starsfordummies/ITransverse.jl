abstract type TrotterScheme end

struct Murg    <: TrotterScheme end
struct SymSVD  <: TrotterScheme end
struct Floquet <: TrotterScheme end

# ── Ising ────────────────────────────────────────────────────────────────────
expH(sites, mp::IsingParams, ::Murg;    dt) = expH_ising_murg(sites, mp; dt)
expH(sites, mp::IsingParams, ::SymSVD;  dt) = expH_ising_symm_svd(sites, mp; dt)
expH(sites, mp::IsingParams, ::Floquet; dt) = expH_ising_floquet(sites, mp; dt)

# ── Potts ────────────────────────────────────────────────────────────────────
expH(sites, mp::PottsParams, ::Murg;   dt) = expH_potts_murg(sites, mp; dt)
expH(sites, mp::PottsParams, ::SymSVD; dt) = expH_potts_symmetric_svd(sites, mp; dt)

# ── XXZ ──────────────────────────────────────────────────────────────────────
expH(sites, mp::XXZParams, ::SymSVD; dt) = expH_XXZ_svd(sites, mp; dt)
