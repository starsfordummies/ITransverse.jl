# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 

abstract type ModelParams end

function ModelParams(phys_space, Jtwo, gperp, hpar)
    @warn "Warning, ModelParams(site, J,g,h) is deprecated - use IsingParams(Jxx, gz, hx) for Ising instead"
    return IsingParams(Jtwo, gperp, hpar)
end

struct IsingParams <: ModelParams
    Jtwo::Float64
    gperp::Float64
    hpar::Float64
    direction::String
end

# Defaults
IsingParams() = IsingParams(1.0, -1.05, 0.5)

IsingParams(Jtwo, gperp, hpar; direction="XXZ") = IsingParams(Jtwo, gperp, hpar, direction)

# allow for changes on the fly of params 
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar, direction=x.direction) = IsingParams(Jtwo, gperp, hpar; direction)

struct PottsParams <: ModelParams
    JSS::Float64
    ftau::Float64
end
