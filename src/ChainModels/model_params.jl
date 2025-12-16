# Custom param structures for models
abstract type ModelParams end


struct NoParams <: ModelParams
    phys_site::Index{Int64}
end


struct IsingParams{T <: Number} <: ModelParams
    Jtwo::T
    gperp::T
    hpar::T
    phys_site::Index{Int64}

end

function IsingParams(Jtwo::Number, gperp::Number, hpar::Number)
    T = promote_type(typeof(Jtwo), typeof(gperp), typeof(hpar))
    IsingParams{T}(T(Jtwo), T(gperp), T(hpar), Index(2, "S=1/2"))
end

Base.:*(dt::Number, mp::IsingParams) = IsingParams(mp.Jtwo * dt, mp.gperp * dt, mp.hpar * dt, mp.phys_site)

# Defaults
IsingParams() = IsingParams(1.0, -1.05, 0.5)


# allow for changes on the fly of params 
IsingParams(x::IsingParams; Jtwo=x.Jtwo, gperp=x.gperp, hpar=x.hpar) = IsingParams(Jtwo, gperp, hpar)


struct PottsParams{T <: Number} <: ModelParams
    JSS::T
    ftau::T
    hS::T
    phys_site::Index{Int64}
end

function  PottsParams(Jtwo::Number, ftau::Number, hpar::Number=0.)
    T = promote_type(typeof(Jtwo), typeof(ftau), typeof(hpar))
    PottsParams{T}(T(Jtwo), T(ftau), T(hpar), Index(3, "S=1"))
end

""" For XXZ We need to specify the physical site, as it can be defined for different spins """

struct XXZParams{T <: Number} <: ModelParams
    J_XY::T
    J_ZZ::T
    hz::T
    phys_site::Index{Int64}
end

function XXZParams(J_XY::Number, J_ZZ::Number, hz::Number=0., phys_site = Index(2, "S=1/2"))
    T = promote_type(typeof(J_XY), typeof(J_ZZ), typeof(hz))
    XXZParams{T}(T(J_XY), T(J_ZZ), T(hz), phys_site)
end


XXZParams(J_ZZ, hz, phys_site) = XXZParams(1, J_ZZ, hz, phys_site)


# Extract only the model parameters in reasonable order 
modelparams(mp::IsingParams) = (mp.Jtwo, mp.gperp, mp.hpar)
modelparams(mp::PottsParams) = (mp.JSS, mp.ftau, mp.hS)
modelparams(mp::XXZParams) = (mp.J_XY, mp.J_ZZ, mp.hz)




""" Model Hamiltonian struct. Contains ModelParams and the H MPO"""
struct ModelHam{T <: ModelParams}
    p::T
    H::MPO
end

function ModelHam(sites, HH::Function, mp::ModelParams)
    return ModelHam(mp, HH(sites::Vector{<:Index}, modelparams(mp)...))
end
