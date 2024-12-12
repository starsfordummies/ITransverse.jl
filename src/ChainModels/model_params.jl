# Custom param structures for Ising-Potts models
# and for power methods/ truncations etc 

abstract type ModelParams end

function ModelParams(phys_space, Jtwo, gperp, hpar)
    @warn "Warning, ModelParams(site, J,g,h) is deprecated - use IsingParams(Jxx, gz, hx) for Ising instead"
    return IsingParams(Jtwo, gperp, hpar)
end

struct IsingParams <: ModelParams
    phys_space::String
    direction::String
    Jtwo::Float64
    gperp::Float64
    hpar::Float64

    IsingParams(Jtwo, gperp, hpar; direction="XXZ") = new("S=1/2", direction, Jtwo, gperp, hpar)
end

# allow for changes on the fly of params 
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar) = IsingParams(Jtwo, gperp, hpar)

