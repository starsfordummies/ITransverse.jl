# Custom param structures for models
abstract type ModelParams end


Base.@kwdef mutable struct NoParams <: ModelParams
    phys_site::Index{Int64} = Index(2, "S=1/2")
end
NoParams(i::Index) = NoParams(; phys_site=i)


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
