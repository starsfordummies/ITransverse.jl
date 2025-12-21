
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
function run_cone(psi::MPS, 
    b::FoldtMPOBlocks,
    cone_pars::ConeParams,
    checkpoint::DoCheckpoint,
    nT_final::Int
)

    (; opt_method, optimize_op, truncp, vwidth) = cone_pars

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = vectorized_identity(dim(b.rot_inds[:R]))

    time_dim = dim(b.WWc,1)

    if truncp.direction == "right"
        sweep_str = "psi0 << Op"
    elseif  truncp.direction == "left"
        sweep_str = "psi0 >> Op"
    else
        sweep_str = "??????"
    end

    nsteps = nT_final - length(psi)

    p = Progress(div(nsteps, vwidth); desc="[cone(v=$vwidth)|$(opt_method)] [$(sweep_str)] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim))", showspeed=true) 

    for nt = 1:nsteps

        # Only extend the cone every vwidth timesteps
        if nt % vwidth == 0

            ts = siteinds(rr)
            #Extend timesites by 1 
            for jj = 1:vwidth
                push!(ts, Index(time_dim, tags="Site,n=$(length(rr)+jj),time_fold"))
            end

            if opt_method == "RTM_LR"
                # if we're worried about symmetry Left-Right, evolve separately L and R 
                rrwork = deepcopy(rr)
                _,rr, ents = extend_tmps_cone(ll, optimize_op, Id, rrwork, ts, b, truncp)
                ll,_, ents = extend_tmps_cone(ll, Id, optimize_op, rrwork, ts, b, truncp)
            elseif opt_method == "RTM_R"
                _,rr, ents = extend_tmps_cone(ll, optimize_op, Id, rr, ts, b, truncp)
                ll = rr
            elseif opt_method == "RTM_L1O1R"
                _,rr, ents = extend_tmps_cone(ll, optimize_op, rr, ts, b, truncp)
                ll = rr
            elseif opt_method == "RTM_SKEW"
                _,rr, ents = extend_tmps_skew(ll, optimize_op, rr, ts, b, truncp)
                ll = rr
            elseif opt_method == "RDM" # TODO Non-symmetric case with RDM ?
                tmpo = folded_tMPO_ext(b, ts; LR=:right, n_ext=vwidth)
                rr = applyn(tmpo,rr; truncate=true, cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                ll = rr
            else
                error("no valid update method specified ($(opt_method))")
            end


            overlapLR = overlap_noconj(ll,rr)

            # At each step we renormalize so that the overlap <L|R>=1 !
            ll *= sqrt(1/overlapLR)
            rr *= sqrt(1/overlapLR)

            state = (L=ll, R=rr, b=b)
            checkpoint(state, nt)


            next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

        end
    end


    return ll, rr, checkpoint
end

