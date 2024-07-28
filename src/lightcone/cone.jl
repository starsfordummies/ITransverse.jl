""" Initializes the light cone folded and rotated temporal MPS |R> given `tMPOParams`
builds a (length n) tMPS with (time_fold)  legs.
Our convention is that after rotation the initial state goes to the *left* 
Returns (psi[the light cone right MPS], b[the folded tMPO building blocks])"""

function init_cone(tp::tMPOParams, n::Int=3)

    b = FoldtMPOBlocks(tp)

    time_dim = dim(b.WWc,1)
    
    ts = [Index(time_dim, tags="Site,n=1,time_fold")]

    psi = folded_right_tMPS(b, ts)

    for jj = 2:n
        push!(ts, Index(time_dim, tags="Site,n=$(jj),time_fold"))
        m = folded_tMPO_R(b,ts)
        psi = apply_extend(m, psi)
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
function apply_extend(A::MPO, ψ::MPS; truncate::Bool=false, cutoff::Float64=1e-14, maxdim::Int=maxlinkdim(A) * maxlinkdim(ψ))

    ψc = deepcopy(ψ)
    push!(ψc.data, ITensor(1))
    ψc = apply(A, ψc; alg="naive", truncate, cutoff, maxdim)
    return ψc
end



"""
One step of the light cone algorithm: takes left and right tMPS ll, rr,
the time MPO and the operator O
extends the time MPO and the left-right tMPS by optimizing 
1) the overlap (L1|OR)  -> save new L
2) the overlap (LO|1R)  -> save new R  (in case non symmetric)

Returns the updated left-right tMPS 
"""
function extend_tmps_cone(ll::MPS, op_L::Vector{<:Number}, op_R::Vector{<:Number}, rr::MPS, 
    ts::Vector{<:Index}, b::FoldtMPOBlocks, truncp::TruncParams)

    @assert length(ts) == length(ll)+1 

    tmpo = folded_tMPO_R(b, ts, op_R)

    psi_R = apply_extend(tmpo, rr)

    tmpo = folded_tMPO_L(b, ts, op_L) 
    # =swapprime(folded_tMPO(tp, time_sites; fold_op=op_L), 0, 1, "Site")

    psi_L = apply_extend(tmpo, ll)

    ll, rr, ents, ov = truncate_rsweep(psi_L,psi_R; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    

    return ll, rr, ents, ov

end




function run_cone(psi::MPS, 
    b::FoldtMPOBlocks,
    cp::ConeParams,
    nsteps::Int
    )

    (; opt_method, optimize_op, which_evs, which_ents, checkpoint, truncp) = cp

    tp = b.tp

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = ComplexF64[1,0,0,1]

    chis = []
    overlaps = []
    times = [] 

    entropies = dictfromlist(which_ents)
    expvals = dictfromlist(which_evs)
    infos = Dict(:times => [], :b => b, :cp => cp)

    time_dim = dim(b.WWc,1)

    p = Progress(nsteps; desc="[cone|$(opt_method)] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim))", showspeed=true) 

    for dt = length(psi):nsteps
        
        ts = siteinds(rr)
        #Extend timesites by 1 
        push!(ts, Index(time_dim, tags="Site,n=$(length(rr)+1),time_fold"))

        if opt_method == "RTM_LR"
            # if we're worried about symmetry, evolve separately L and R 
            rrwork = deepcopy(rr)
            _,rr, ents = extend_tmps_cone(ll, optimize_op, Id, rrwork, ts, b, truncp)
            ll,_, ents = extend_tmps_cone(ll, Id, optimize_op, rrwork, ts, b, truncp)
        elseif opt_method == "RTM_R"
            _,rr, ents = extend_tmps_cone(ll, optimize_op, Id, rr, ts, b, truncp)
            ll = rr
        elseif opt_method == "RDM" # TODO Non-symmetric case with RDM ?
            tmpo = folded_tMPO_R(b, ts)
            rr = apply_extend(tmpo,rr; truncate=true, cutoff=truncp.cutoff, maxdim=truncp.maxbondim)
            ll = rr
        else
            @error "no valid update method specified"
            @error "RTM_LR|RTM_R|RDM"
        end


        overlapLR = overlap_noconj(ll,rr)

        # At each step we renormalize so that the overlap <L|R>=1 !
        ll *= sqrt(1/overlapLR)
        rr *= sqrt(1/overlapLR)

        evs_computed = compute_expvals(ll, rr, which_evs, b)

        mergedicts!(expvals, evs_computed)

        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)
        push!(times, length(ll)*tp.dt)


        #ent = vn_entanglement_entropy(ll)
        #TODO Compute entropies

        if checkpoint > 0 && length(ll) > 50 && length(ll) % checkpoint == 0
            llcp = tocpu(ll)
            rrcp = tocpu(rr)
            jldsave("cp_cone_$(length(ll))_chi_$(chis[end]).jld2"; llcp, rrcp, chis, times, expvals, entropies, infos)
        end

        next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end


    return ll, rr, chis, expvals, entropies, infos
end


# TODO need to update it to newest 
#=
""" Resumes a light cone simulation from a checkpoint file """
function resume_cone(checkpoint::String, nsteps::Int)

    c = jldopen(checkpoint, "r")

    psi = c["rrcp"]
    cp = c["infos"][:cp]
    tp = c["infos"][:tp]

    # TODO extend with prev results 
    return run_cone(psi, nsteps, op, tp, truncp)
    
end
=#
