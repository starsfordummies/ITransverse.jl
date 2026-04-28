
""" Given an MPO A and a MPS ψ, with length(A) = length(ψ)+1, 
Extends MPS ψ to the *right* by one site by applying the MPO,
Returns a new MPS which is the extension of ψ, with siteinds matching those of A.
In its current version, we allow to apply an MPO with a two-legged tensor at its right edge,
which I think only works with the "naive" algorithm. We don't perform any truncation here 

```
        | | | | | |  |
        o-o-o-o-o-o--o
        | | | | | |  
        o-o-o-o-o-o
``` 
"""
function run_cone(ll::MPS, rr::MPS,
    b::FoldtMPOBlocks,
    cone_pars::ConeParams,
    checkpoint::DoCheckpoint,
    nT_final::Int
)

    (; opt_method, optimize_op, truncp, vwidth) = cone_pars

    Id = vectorized_identity(dim(b.iR))

    time_dim = dim(b.WWc,1)

    if truncp.direction == :right
        sweep_str = "ψ0<=Op"
    elseif  truncp.direction == :left
        sweep_str = "ψ0=>Op"
    else
        sweep_str = "???"
    end

    start_length = length(rr)
    @assert length(ll) == length(rr)

    nsteps = div(nT_final - length(rr), vwidth)

    info_str = "[cone(v=$vwidth)|$(opt_method)|$(truncp.alg)] [$(sweep_str)] cutoff=$(truncp.cutoff), maxdim=$(truncp.maxdim))"
    p = Progress(nsteps; desc=info_str, showspeed=true) 

    time_steps = (start_length + vwidth) : vwidth : nT_final

    for nt in time_steps

        ts = siteinds(rr)
        n_ext = nt - length(rr)
        append!(ts, [Index(time_dim, tags="Site,n=$(length(rr)+jj),time_fold") for jj in 1:n_ext])

        ll, rr, sv = if opt_method == :sym

            tmpoL = folded_tMPO_ext(b, ts; LR=:left, fold_op=optimize_op, n_ext) 
            tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=Id, n_ext)
        
            _, rr, sv = tlrapply(ll, tmpoL, tmpoR, rr; truncp...)

            sim(linkinds, rr), rr, sv
            
        else # update both 

            rrp = copy(rr)

            tmpoL = folded_tMPO_ext(b, ts; LR=:left,  fold_op=optimize_op, n_ext) 
            tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=Id,          n_ext)
        
            _, rr, _ = tlrapply(ll, tmpoL, tmpoR, rrp; truncp...)

            tmpoL = folded_tMPO_ext(b, ts; LR=:left,  fold_op=Id,          n_ext) 
            tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=optimize_op, n_ext)
        
            ll, _, sv = tlrapply(ll, tmpoL, tmpoR, rrp; truncp...)

            ll, rr, sv
        end


        overlapLR = overlap_noconj(ll,rr)

        # At each step we renormalize so that the overlap <L|R>=1 !
        ll *= sqrt(1/overlapLR)
        rr *= sqrt(1/overlapLR)

        state = (L=ll, R=rr, b=b)
        checkpoint(state, nt)


        next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end

    write_cp(checkpoint; filename="OUTcone_final.jld2")
    return ll, rr, checkpoint
end

""" Single-MPS convenience overload: ll and rr both start as deep copies of `psi`. """
function run_cone(psi::MPS,
    b::FoldtMPOBlocks,
    cone_pars::ConeParams,
    checkpoint::DoCheckpoint,
    nT_final::Int
)
    run_cone(copy(psi), copy(psi), b, cone_pars, checkpoint, nT_final)
end

"""
    resume_cone(cp::DoCheckpoint, cone_pars::ConeParams, nT_final::Int)

Load the latest snapshot stored in `cp` and continue running the cone up to
`nT_final` total time steps.  The snapshot must contain `L`, `R`, and `b`
(i.e. `f_savestate` must have been configured with those three keys when the
original `DoCheckpoint` was created).

Previously recorded steps and observables are preserved; new data are appended.
"""
function resume_cone(cp::DoCheckpoint, nT_final::Int; 
    cone_pars::ConeParams=cp.params["cparams"],
    do_gpu::Bool=false)

    latest = cp.latest
 
    ll, rr, b = if do_gpu 
        togpu(latest.L), togpu(latest.R), togpu(latest.b)
    else
        latest.L, latest.R, latest.b
    end

    nT_start = length(ll) 
    @info "resume_cone: resuming from step $nT_start → $nT_final  (length(L)=$(length(ll)))"

    return run_cone(ll, rr, b, cone_pars, cp, nT_final)
end

"""
    resume_cone(filename::String, cone_pars::ConeParams, nT_final::Int; kwargs...)

Convenience overload: load the checkpoint file `filename`, reconstruct a
`DoCheckpoint` with the same `f_obs` / `f_savestate` supplied via `kwargs`,
and resume the cone.  Pass at minimum `f_obs` and `f_savestate` matching the
original run if you want observables to keep being recorded.
"""
function resume_cone(filename::String, nT_final::Int; 
                     f_obs=NamedTuple(), f_savestate=NamedTuple(), kwargs...)

    params  = load(filename, "params")  

    steps   = load(filename, "steps")
    obs_hist = load(filename, "observables")
    latest  = load(filename, "latest")
    save_at = load(filename, "save_at")

    cp = DoCheckpoint(
        filename;
        params,
        save_at,
        f_obs,
        f_savestate,
        steps,
        obs_hist,
        latest,
    )

    return resume_cone(cp, nT_final; kwargs...)
end

