
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

    if truncp.direction == :right
        sweep_str = "ψ0<=Op"
    elseif  truncp.direction == :left
        sweep_str = "ψ0=>Op"
    else
        sweep_str = "???"
    end

    nsteps = nT_final - length(psi)

    p = Progress(div(nsteps, vwidth); desc="[cone(v=$vwidth)|$(opt_method)|$(truncp.alg)] [$(sweep_str)] $cutoff=$(truncp.cutoff), maxdim=$(truncp.maxdim))", showspeed=true) 

    for nt = 1:nsteps

        # Only extend the cone every vwidth timesteps
        if nt % vwidth == 0

            ts = siteinds(rr)
            #Extend timesites by 1 
            for jj = 1:vwidth
                push!(ts, Index(time_dim, tags="Site,n=$(length(rr)+jj),time_fold"))
            end

            # We can extend by more than one timestep if the cone is narrow
            n_ext = length(ts) - length(ll)

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
    end


    return ll, rr, checkpoint
end

