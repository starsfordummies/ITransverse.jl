""" Initializes the light cone folded and rotated temporal MPS |R> given `tMPOParams`
builds a (length n) tMPS with (time_fold)  legs.
Returns (psi[the light cone right MPS], b[the folded tMPO building blocks])"""
function init_cone(tp::tMPOParams, n::Int=3)
    b = FoldtMPOBlocks(tp)
    init_cone(b, n)
end

function init_cone(b::FoldtMPOBlocks, n::Int=3)

    time_dim = dim(b.WWc,1)
    
    ts = [Index(time_dim, tags="Site,n=1,time_fold")]

    psi = folded_right_tMPS(b, ts)

    for jj = 2:n
        push!(ts, Index(time_dim, tags="Site,n=$(jj),time_fold"))
        m = folded_tMPO_ext(b,ts; LR="R")
        psi = applyn(m, psi)
    end

    return psi, b
end



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
    cp::ConeParams,
    nT_final::Int
    )

    (; opt_method, optimize_op, which_evs, which_ents, checkpoint, truncp, vwidth) = cp

    fn_cp = nothing

    tp = b.tp

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = vectorized_identity(dim(b.rot_inds[:R]))

    chis = Int[]
    overlaps = ComplexF64[]
    times = typeof(b.tp.dt)[]

    entropies = dictfromlist(which_ents)
    expvals = dictfromlist(which_evs)

    # For checkpoints, we want to save CPU data 
    tp_cp =  tMPOParams(tp; bl=tocpu(tp.bl))
    
    infos = Dict(:times => times, :tp => tp_cp, :coneparams => cp, :dtype => NDTensors.unwrap_array_type(tp.bl))

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
            elseif opt_method == "RDM" # TODO Non-symmetric case with RDM ?
                tmpo = folded_tMPO_ext(b, ts; LR="R", n_ext=vwidth)
                rr = applyn(tmpo,rr; truncate=true, cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                ll = rr
            else
                error("no valid update method specified ($(opt_method)): use RTM_LR|RTM_R|RDM")
            end


            overlapLR = overlap_noconj(ll,rr)

            # At each step we renormalize so that the overlap <L|R>=1 !
            ll *= sqrt(1/overlapLR)
            rr *= sqrt(1/overlapLR)

            evs_computed = compute_expvals(ll, rr, which_evs, b)

            mergedicts!(expvals, evs_computed)

            #@show evs_computed
            #@show expvals

            push!(chis, maxlinkdim(ll))
            push!(overlaps, overlapLR)
            push!(times, length(ll)*tp.dt)


            #ent = vn_entanglement_entropy(ll)
            #TODO Compute entropies
            if haskey(entropies, "VN")
                push!(entropies["VN"], vn_entanglement_entropy(rr))
            end
            if haskey(entropies, "GENR2")
                push!(entropies["GENR2"], rtm2_contracted(ll,rr))
            end
            if haskey(entropies, "GENR2_Pz")
                tmpo = folded_tMPO(b, ts, fold_op=[1,0,0,0])
                rr2 = apply(tmpo, rr;  cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                ll2 = applys(tmpo, ll; cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                for jj = 1:7
                    rr2 = apply(tmpo, rr2;  cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                    ll2 = applys(tmpo, ll2; cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                end
                push!(entropies["GENR2_Pz"], rtm2_contracted(ll2,rr2))
            end
            if haskey(entropies, "GENVN_Pz")
                tmpo = folded_tMPO(b, ts, fold_op=[1,0,0,0])
                rr2 = apply(tmpo, rr;  cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                for jj = 1:11
                    rr2 = apply(tmpo, rr2;  cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                    #ll2 = applys(tmpo, ll; cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                end
                #rr2 = apply(tmpo, rr; cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
                push!(entropies["GENVN_Pz"], generalized_vn_entropy_symmetric(rr2))
            end
            if haskey(entropies, "GENVN")
                push!(entropies["GENVN"], generalized_vn_entropy_symmetric(ll))
            end


            if checkpoint > 0 && length(ll) > 50 && length(ll) % checkpoint == 0
                fn_cp = "cp_cone_$(length(ll))_chi_$(chis[end]).jld2"
                infos[:times] = length(ll) - length(expvals) : length(ll)
                llcp = tocpu(ll)
                rrcp = tocpu(rr)
                jldsave(fn_cp; llcp, rrcp, chis, times, expvals, entropies, infos)
            end

            next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

        end
    end


    return ll, rr, chis, expvals, entropies, infos, fn_cp
end

