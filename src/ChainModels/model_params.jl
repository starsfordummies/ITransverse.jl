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
    phys_space::String

    function IsingParams(Jtwo::Number, gperp::Number, hpar::Number, direction::String="XXZ")
        new(Float64(Jtwo), Float64(gperp), Float64(hpar), direction, "S=1/2")
    end
end

# Defaults
IsingParams() = IsingParams(1.0, -1.05, 0.5)

# IsingParams(Jtwo, gperp, hpar; direction="XXZ") = IsingParams(Jtwo, gperp, hpar, direction)


# allow for changes on the fly of params 
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar, direction=x.direction) = IsingParams(Jtwo, gperp, hpar; direction)

struct PottsParams <: ModelParams
    JSS::Float64
    ftau::Float64
    hS::Float64
    phys_space::String
    function PottsParams(Jtwo::Number, ftau::Number, hpar::Number=0.)
        new(Float64(Jtwo), Float64(ftau), Float64(hpar), "S=1")
    end
end

struct XXZParams <: ModelParams
    J_XY::Float64
    J_ZZ::Float64
    hz::Float64
    phys_space::String

    function XXZParams(J_XY::Number, J_ZZ::Number, hz::Number=0.)
        new(Float64(J_XY), Float64(J_ZZ), Float64(hz), "S=1/2")
    end
end

struct XXZSpin1Params <: ModelParams
    J_XY::Float64
    J_ZZ::Float64
    hz::Float64
    phys_space::String

    function XXZSpin1Params(J_XY::Number, J_ZZ::Number, hz::Number=0.)
        new(Float64(J_XY), Float64(J_ZZ), Float64(hz), "S=1")
    end
end

XXZParams(J_ZZ,hz) = XXZParams(1, J_ZZ, hz)
XXZSpin1Params(J_ZZ,hz) = XXZSpin1Params(1, J_ZZ, hz)
