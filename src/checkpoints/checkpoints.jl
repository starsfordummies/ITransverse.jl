mutable struct DoCheckpoint{TParams, TFObs, TFState, TSaveAt}
    filename::String  # output CP file 
    save_at::TSaveAt  # any iterable of steps at which to save
    params::TParams # any params one wishes to save 
    steps::Vector{Int}
    f_obs::TFObs
    obs_hist::Dict{Symbol, Vector}
    f_savestate::TFState
    latest::Union{Nothing,NamedTuple}
end


""" Initialize CP """
function DoCheckpoint(filename;
                      params,
                      save_at=Int[],
                      f_obs=NamedTuple(),
                      f_savestate=NamedTuple())

    obs_hist = Dict{Symbol, Vector}()
    @info "CP: Initializing observables $(keys(f_obs))"
    for name in keys(f_obs)
        obs_hist[name] = Any[]
    end

    DoCheckpoint(
        filename,
        save_at,
        params,
        Int[],
        f_obs,
        obs_hist,
        f_savestate,
        nothing  # empty snapshot
    )
end

""" Write the current cp state to disk. """
function write_cp(cp::DoCheckpoint; filename=cp.filename)
    for (k, v) in pairs(cp.obs_hist)
        cp.obs_hist[k] = collect(promote(v...))
    end
    @info "Saving CP $(cp.filename)..."
    save(filename,
         "params",      cp.params,
         "steps",       cp.steps,
         "save_at",     cp.save_at,
         "observables", cp.obs_hist,
         "latest",      cp.latest)
end

""" Saves checkpoint """
function (cp::DoCheckpoint)(state, step::Int)

    # history observables
    for (name, obs) in pairs(cp.f_obs)
        push!(cp.obs_hist[name], obs(state))
    end

    # build latest snapshot
    cp.latest = NamedTuple(
        name => tocpu(f(state)) for (name, f) in pairs(cp.f_savestate)
    )

    push!(cp.steps, length(cp.latest.R))

    if step in cp.save_at
        write_cp(cp)
    end
end
