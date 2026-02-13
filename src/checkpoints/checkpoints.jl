""" Build list of steps at which we should checkpoint """
function which_cps(checkpoints)
    # Checkpoints logic, let's try and be flexible 
    checkpoints = if isa(checkpoints, Integer)
        if checkpoints > 0 
            collect(50:checkpoints:20000)         
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
    filename::String
    save_at::Vector{Int}
    params::TParams
    steps::Vector{Int}
    observables::TObs
    history::Dict{Symbol, Vector}
    latest_savers::TLatestFns
    latest::Union{Nothing,NamedTuple}
end


""" Initialize CP """
function DoCheckpoint(filename;
                      params,
                      save_at=Int[],
                      observables=NamedTuple(),
                      latest_savers=NamedTuple())

    history = Dict{Symbol, Vector}()
    @info "CP: Initializing observables $(keys(observables))"
    for name in keys(observables)
        history[name] = Any[]
    end

    DoCheckpoint(
        filename,
        which_cps(save_at),
        params,
        Int[],
        observables,
        history,
        latest_savers,
        nothing  # empty snapshot
    )
end

""" Saves checkpoint """
function (cp::DoCheckpoint)(state, step::Int)
    push!(cp.steps, step)

    # history observables
    for (name, obs) in pairs(cp.observables)
        push!(cp.history[name], obs(state))
    end

    # build latest snapshot
    cp.latest = NamedTuple(
        name => tocpu(f(state)) for (name, f) in pairs(cp.latest_savers)
    )

    # TODO  PROMOTE ANYs  # v2 = collect(promote(v...))
    # TODO CONVERT to CPU the state 
    if step in cp.save_at

        for (k,v) in pairs(cp.history)
            cp.history[k] = collect(promote(v...))
        end
        @info "Step $(step): Saving CP $(cp.filename)..." 
        save(cp.filename,
             "steps", cp.steps,
             "observables", cp.history,
             "latest", cp.latest)
    end
end
