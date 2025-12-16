# Custom param structures for models


abstract type ModelParams end

struct NoParams <: ModelParams
    phys_site::Index{Int64}
end

struct IsingParams <: ModelParams
    Jtwo::Float64
    gperp::Float64
    hpar::Float64
    phys_site::Index{Int64}

    function IsingParams(Jtwo::Number, gperp::Number, hpar::Number)
        new(Float64(Jtwo), Float64(gperp), Float64(hpar), Index(2, "S=1/2"))
    end
end

# Defaults
IsingParams() = IsingParams(1.0, -1.05, 0.5)


# allow for changes on the fly of params 
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar) = IsingParams(Jtwo, gperp, hpar)


struct PottsParams <: ModelParams
    JSS::Float64
    ftau::Float64
    hS::Float64
    phys_site::Index{Int64}
    function PottsParams(Jtwo::Number, ftau::Number, hpar::Number=0.)
        new(Float64(Jtwo), Float64(ftau), Float64(hpar), Index(3, "S=1"))
    end
end

""" For XXZ We need to specify the physical site, as it can be defined for different spins """

struct XXZParams <: ModelParams
    J_XY::Float64
    J_ZZ::Float64
    hz::Float64
    phys_site::Index{Int64}

    function XXZParams(J_XY::Number, J_ZZ::Number, hz::Number=0., phys_site = Index(2, "S=1/2"))
        new(Float64(J_XY), Float64(J_ZZ), Float64(hz), phys_site)
    end
end


XXZParams(J_ZZ,hz, phys_site) = XXZParams(1, J_ZZ, hz, phys_site)


# Extract only the model parameters in reasonable order 
modelparams(mp::IsingParams) = (mp.Jwo, mp.gperp, mp.hpar)
modelparams(mp::PottsParams) = (mp.JSS, mp.ftau, mp.hS)
modelparams(mp::XXZParams) = (mp.J_XY, mp.J_ZZ, mp.hz)




""" Model Hamiltonian struct. Contains ModelParams and the H MPO"""
struct ModelHam{T <: ModelParams}
    p::T
    H::MPO
end

function ModelHam(sites, HH::Function, mp::ModelParams)
    return ModelHam(mp, HH(sites::Vector{<:Index}, modelparams(mp)))
end
