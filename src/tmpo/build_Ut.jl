""" Queen of all boilerplate """
function _build_Ut(sites::Vector{<:Index}, 
    fUt::Function, model_params...; dt::Number, build_4o::Bool)
    Ut = if build_4o
        p = 1.0/(4- 4^(1/3))
        U1 = fUt(sites, model_params...; dt=p*dt)
        U2 = fUt(sites, model_params...; dt=(1-4p)*dt)

        UU1 = applyn(U1,U1)
        U4 = applyn(U2, UU1)
        U4 = applyn(UU1, U4)
 
    else
        fUt(sites, model_params...; dt)
    end

    return Ut 
end

function build_Ut(sites::Vector{<:Index}, fUt::Function, mp::ModelParams; dt::Number, build_4o::Bool=false)
    _build_Ut(sites, fUt, modelparams(mp)...; dt, build_4o)
end

function build_Ut(sites::Vector{<:Index}, tp::tMPOParams; dt::Number=tp.dt, build_4o::Bool=false)
    build_Ut(sites, tp.expH_func, tp.mp; dt, build_4o)
end

function build_Ut(fUt::Function, mp::ModelParams; kwargs...)
    ss = [addtags(sim(mp.phys_site),"Site") for _ in 1:3]
    build_Ut(ss, fUt, mp; kwargs...)
end

function build_Ut(tp::tMPOParams; dt::Number=tp.dt, kwargs...)
    build_Ut(tp.expH_func, tp.mp; dt, kwargs...)
end

function build_Ut(b::FwtMPOBlocks; kwargs...)
    build_Ut(b.tp; kwargs...)
end
