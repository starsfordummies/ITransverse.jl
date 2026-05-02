""" Queen of all boilerplate """
function _build_Ut(sites::Vector{<:Index}, 
    scheme::ExpHRecipe, mp::ModelParams; dt::Number, build_4o::Bool)
    if build_4o
        p = 1.0/(4 - 4^(1/3))
        U1 = expH(sites, mp, scheme; dt=p*dt)
        U2 = expH(sites, mp, scheme; dt=(1-4p)*dt)
        UU1 = applyn(U1, U1)
        return applyn(UU1, applyn(U2, UU1))
    else
        return expH(sites, mp, scheme; dt)
    end
end

function build_Ut(sites::Vector{<:Index}, scheme::ExpHRecipe, mp::ModelParams; dt::Number, build_4o::Bool=false)
    _build_Ut(sites, scheme, mp; dt, build_4o)
end

function build_Ut(sites::Vector{<:Index}, tp::tMPOParams; dt::Number=tp.dt, build_4o::Bool=false)
    Ut = build_Ut(sites, tp.scheme, tp.mp; dt, build_4o)
    return adapt(NDTensors.unwrap_array_type(tp.bl), Ut)
end

function build_Ut(scheme::ExpHRecipe, mp::ModelParams; kwargs...)
    ss = [addtags(sim(mp.phys_site),"Site") for _ in 1:3]
    build_Ut(ss, scheme, mp; kwargs...)
end

function build_Ut(tp::tMPOParams; dt::Number=tp.dt, kwargs...)
    Ut = build_Ut(tp.scheme, tp.mp; dt, kwargs...)
    return adapt(NDTensors.unwrap_array_type(tp.bl), Ut)
end

function build_Ut(b::FwtMPOBlocks; kwargs...)
    build_Ut(b.tp; kwargs...)
end
