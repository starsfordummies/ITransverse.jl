
# Truncation algorithms that act on one vector alone, with no left-right environment
const _ONE_SIDED_ALGS = ("densitymatrix", "naive", "cudensitymatrix")

"""
    sym_left(rr::MPS)

The left vector of a left-right symmetric network: `rr` transposed (QN arrows reversed, see
[`transpose_arrows`](@ref)), with its own link indices. Unlike `transpose_arrows` it never
copies: every tensor is a view of the data of `rr`, so it costs no memory. Do not modify it
in place.
"""
function sym_left(rr::MPS)
    ll = sim(linkinds, rr)  # new link Indices, same storage
    hasqns(ll) || return ll
    for j in eachindex(ll)
        ll.data[j] = ITensors.setinds(ll[j], dag(inds(ll[j])))  # arrows flipped, same storage
    end
    return ll
end

""" Runs the light cone algorithm up to a length of nT_final timesteps
"""
function run_cone(ll::TMPSorMPS, rr::TMPSorMPS,
    b::FoldtMPOBlocks,
    cone_pars::ConeParams,
    checkpoint::DoCheckpoint,
    nT_final::Int
)
    ll, rr = unsided(ll), unsided(rr)  # accept a tagged boundary vector, work on the MPS

    (; opt_method, optimize_op, truncp, vwidth) = cone_pars

    Id = vectorized_identity(dim(b.iR))

    # Symmetric case with an algorithm that truncates each vector on its own: the left vector is
    # just the transpose of the right one, so only the right one is built and truncated
    one_sided = opt_method == :sym && truncp.alg in _ONE_SIDED_ALGS

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
        append!(ts, [sim(b.iP; tags="Site,n=$(length(rr)+jj),time_fold") for jj in 1:n_ext])

        ll, rr, sv = if one_sided

            tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=Id, n_ext)
            ll = nothing  # drop the reference to the old view, so its R can be freed

            rr, sv = tapply(tmpoR, rr; truncp...)

            nothing, rr, sv

        elseif opt_method == :sym

            tmpoL = folded_tMPO_ext(b, ts; LR=:left, fold_op=optimize_op, n_ext) 
            tmpoR = folded_tMPO_ext(b, ts; LR=:right, fold_op=Id, n_ext)
        
            _, rr, sv = tlrapply(ll, tmpoL, tmpoR, rr; truncp...)

            nothing, rr, sv
            
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


        # At each step we renormalize so that the overlap <L|R>=1 !
        overlapLR = if opt_method == :sym
            ov = overlap_noconj(sym_left(rr), rr)
            rr *= sqrt(1/ov)
            ll = sym_left(rr)  # L = R^T, so <L|R> picks up the factor twice
            ov
        else
            ov = overlap_noconj(ll,rr)
            ll *= sqrt(1/ov)
            rr *= sqrt(1/ov)
            ov
        end

        state = (L=ll, R=rr, b=b, sv=sv)  # sv is TruncLR.sv: χ x ncuts SVD singular values matrix
        checkpoint(state, nt)


        next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end

    write_cp(checkpoint; filename="OUTcone_final.jld2")
    return ll, rr, checkpoint
end

""" Single-MPS convenience overload: ll and rr both start as deep copies of `psi`. """
function run_cone(psi::TMPSorMPS,
    b::FoldtMPOBlocks,
    cone_pars::ConeParams,
    checkpoint::DoCheckpoint,
    nT_final::Int
)
    psi = unsided(psi)  # accept a tagged boundary vector, work on the MPS
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

