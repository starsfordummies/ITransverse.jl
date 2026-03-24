""" Horrible boilerplate code building a list of steps at which we should checkpoint """
function which_cps(checkpoints)
    # Checkpoints logic, let's try and be flexible 
    checkpoints = if isa(checkpoints, Integer)
        if checkpoints > 0 
            collect(50:checkpoints:10000)         
        else
            Int[]
        end
    elseif isa(checkpoints, Tuple{Int})
        checkpoints                              # tuple → keep as is
    elseif isa(checkpoints, AbstractVector{Int})
        collect(checkpoints) 
    elseif isa(checkpoints, AbstractRange{Int})
        collect(checkpoints)                    # range → tuple
    else
        throw(ArgumentError("Unsupported input type $(typeof(checkpoints))"))
    end

    return checkpoints
end


mutable struct DoCheckpoint{TParams, TObs, TLatestFns}
    filename::String  # output CP file 
    save_at::Vector{Int} # at which step we save
    params::TParams
    steps::Vector{Int}
    f_obs::TObs
    obs_hist::Dict{Symbol, Vector}
    latest_savers::TLatestFns
    latest::Union{Nothing,NamedTuple}
end


""" Initialize CP """
function DoCheckpoint(filename;
                      params,
                      save_at=Int[],
                      f_obs=NamedTuple(),
                      latest_savers=NamedTuple())

    obs_hist = Dict{Symbol, Vector}()
    @info "CP: Initializing observables $(keys(f_obs))"
    for name in keys(f_obs)
        obs_hist[name] = Any[]
    end

    DoCheckpoint(
        filename,
        which_cps(save_at),
        params,
        Int[],
        f_obs,
        obs_hist,
        latest_savers,
        nothing  # empty snapshot
    )
end

""" Saves checkpoint """
function (cp::DoCheckpoint)(state, step::Int)
    push!(cp.steps, step)

    # history observables
    for (name, obs) in pairs(cp.f_obs)
        push!(cp.obs_hist[name], obs(state))
    end

    # build latest snapshot
    cp.latest = NamedTuple(
        name => tocpu(f(state)) for (name, f) in pairs(cp.latest_savers)
    )

    # TODO  PROMOTE ANYs  # v2 = collect(promote(v...))
    # TODO CONVERT to CPU the state 
    if step in cp.save_at

        for (k,v) in pairs(cp.obs_hist)
            cp.history[k] = collect(promote(v...))
        end
        @info "Step $(step): Saving CP $(cp.filename)..." 
        save(cp.filename,
             "steps", cp.steps,
             "observables", cp.obs_hist,
             "latest", cp.latest)
    end
end
