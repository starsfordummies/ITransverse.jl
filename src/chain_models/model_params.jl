# Custom param structures for models

""" Abstract supertype for all chain-model parameter structs.
Concrete subtypes: [`IsingParams`](@ref), [`PottsParams`](@ref),
[`XXZParams`](@ref), [`NoParams`](@ref).
"""
abstract type ModelParams end


""" Placeholder model parameters with no physical couplings.
Used when the time-evolution tensors are supplied externally (e.g. Floquet gates).

Fields: `phys_site` – the physical site index (default `S=1/2`).
"""
Base.@kwdef mutable struct NoParams <: ModelParams
    phys_site::Index{Int64} = Index(2, "S=1/2")
end
NoParams(i::Index) = NoParams(; phys_site=i)


"""
    IsingParams(Jtwo, gperp, hpar=0; phys_site=Index(2,"S=1/2"))

Parameters for the transverse-field Ising chain
``H = -J_{\\!\\rm two}\\sum_i X_i X_{i+1} - g_\\perp \\sum_i Z_i - h_\\parallel \\sum_i X_i``.

Fields:
- `Jtwo`   – nearest-neighbour XX coupling (default 1.0)
- `gperp`  – transverse field (default 0.4)
- `hpar`   – longitudinal field (default 0.0)
- `phys_site` – physical site index
"""
Base.@kwdef mutable struct IsingParams <: ModelParams
    Jtwo::Float64  = 1.0
    gperp::Float64 = 0.4
    hpar::Float64  = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end
# Positional constructor for backward compatibility: IsingParams(J, g, h)
IsingParams(Jtwo::Number, gperp::Number, hpar::Number) =
    IsingParams(; Jtwo=Float64(Jtwo), gperp=Float64(gperp), hpar=Float64(hpar))
# Copy constructor
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar) =
    IsingParams(; Jtwo, gperp, hpar)


"""
    PottsParams(JSS, ftau; phys_site=Index(3,"S=1"))

Parameters for the 3-state Potts chain.

Fields:
- `JSS`   – spin-spin coupling (default 1.0)
- `ftau`  – transverse (clock) field (default 0.4)
- `phys_site` – physical site index (default `S=1`, i.e. 3-dimensional)
"""
Base.@kwdef mutable struct PottsParams <: ModelParams
    JSS::Float64  = 1.0
    ftau::Float64 = 0.4
    phys_site::Index{Int64} = Index(3, "S=1")
end
# Positional constructor for backward compatibility: PottsParams(J, f)
PottsParams(JSS::Number, ftau::Number) =
    PottsParams(; JSS=Float64(JSS), ftau=Float64(ftau))
# Copy constructor
PottsParams(x::PottsParams; JSS=x.JSS, ftau=x.ftau) =
    PottsParams(; JSS, ftau)


"""
    XXZParams(J_XY, J_ZZ, hz=0; phys_site=Index(2,"S=1/2"))

Parameters for the XXZ spin-1/2 chain
``H = J_{XY}(S^+_i S^-_{i+1} + \\text{h.c.}) + J_{ZZ} S^z_i S^z_{i+1} - h_z \\sum_i S^z_i``.

Fields:
- `J_XY` – XY coupling (default 1.0)
- `J_ZZ` – Ising (ZZ) anisotropy (default 1.0)
- `hz`   – longitudinal field (default 0.0)
- `phys_site` – physical site index
"""
Base.@kwdef mutable struct XXZParams <: ModelParams
    J_XY::Float64  = 1.0
    J_ZZ::Float64  = 1.0
    hz::Float64    = 0.0
    phys_site::Index{Int64} = Index(2, "S=1/2")
end
# Positional constructors for backward compatibility
XXZParams(J_XY::Number, J_ZZ::Number, hz::Number=0.0) =
    XXZParams(; J_XY=Float64(J_XY), J_ZZ=Float64(J_ZZ), hz=Float64(hz))
XXZParams(J_XY::Number, J_ZZ::Number, hz::Number, phys_site::Index) =
    XXZParams(; J_XY=Float64(J_XY), J_ZZ=Float64(J_ZZ), hz=Float64(hz), phys_site)
# Copy constructor
XXZParams(x::XXZParams; J_XY=x.J_XY, J_ZZ=x.J_ZZ, hz=x.hz) =
    XXZParams(; J_XY, J_ZZ, hz, phys_site=x.phys_site)
