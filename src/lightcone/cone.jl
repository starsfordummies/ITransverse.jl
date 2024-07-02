""" Initializes the light cone folded and rotated temporal MPS |R> given `tmpo_params`
builds a (length n) tMPS with a single tensor with one (time,physical) open leg.
After rotation, initial state goes to the *left*."""

function init_cone(tp::tmpo_params, n::Int=3)

    b = FoldtMPOBlocks(tp)

    time_dim = dim(b.WWc,1)
    
    ts = [Index(time_dim, tags="Site,n=1,time_fold")]

    psi = folded_right_tMPS(b, ts)

    for jj = 2:n
        push!(ts, Index(time_dim, tags="Site,n=$(jj),time_fold"))
        m = folded_tMPO_R(b,ts)
        psi = apply_extend(m, psi)
    end

    return psi
end


""" Given an MPO A and a MPS ψ with length(A) = length(ψ)+1, 
Extends MPS ψ to the *right* by one site by applying the MPO,
Returns a new MPS which is the extension of ψ, with siteinds matching those of A
```
        | | | | | |  |
 [rho0]-o-o-o-o-o-o--o--[op]
        | | | | | |  V
        o-o-o-o-o-o
``` 
"""
function apply_extend(A::MPO, ψ::MPS)

    ψc = deepcopy(ψ)
    push!(ψc.data, ITensor(1))
    ψc = applyn(A, ψc)
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
function extend_tmps_cone(rr::MPS, ll::MPS, 
    op_R::Vector{<:Number}, op_L::Vector{<:Number}, 
    ts::Vector{<:Index},
    b::FoldtMPOBlocks,
    truncp::trunc_params,
    compute_r2::Bool=false)

    tmpo = folded_tMPO_R(b, ts, op_R)

    psi_R = apply_extend(tmpo, rr)

    tmpo = folded_tMPO_L(b, ts, op_L) 
    # =swapprime(folded_tMPO(tp, time_sites; fold_op=op_L), 0, 1, "Site")

    psi_L = apply_extend(tmpo, ll)

    rr, ll, ents = truncate_rsweep(psi_R,psi_L; cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    
    gen_renyi2 = [0.]
    if compute_r2
        gen_renyi2 = generalized_renyi_entropy(ll, rr, 2, normalize=true)
    end

    return rr, ll, gen_renyi2 # ents

end





function run_cone(psi::MPS, 
    nsteps::Int, 
    op::Vector{ComplexF64}, 
    tp::tmpo_params,
    truncp::trunc_params,
    save_cp::Bool=true
    )

    ll = deepcopy(psi)
    rr = deepcopy(psi)

    Id = ComplexF64[1,0,0,1]

    chis = []
    overlaps = []
    vn_ents = []
    gen_r2sL = []
    gen_r2sR = []
    times = [] 

    entropies = Dict(:genr2L => gen_r2sL, :genr2R => gen_r2sR, :vn => vn_ents)

    which_evs = ["X","Z","eps"]
    expvals = Dict()
    for op in which_evs
        expvals[op] = []
    end


    infos = Dict(:times => times, :truncp => truncp, :tp => tp, :op => op)

    b = FoldtMPOBlocks(tp)
    time_dim = dim(b.WWc,1)


    p = Progress(nsteps; desc="[cone] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim)), method=$(truncp.ortho_method)", showspeed=true) 

    for dt = length(psi):nsteps
        
        # Original should work 
        llwork = deepcopy(ll)

        ts = siteinds(ll)
        push!(ts, Index(time_dim, tags="Site,n=$(length(ll)+1),time_fold"))

        # if we're worried about symmetry, evolve separately L and R 
        ll,_, ents = extend_tmps_cone(llwork, rr, Id, op, ts, b, truncp)
        push!(gen_r2sL, ents)

        _,rr, ents = extend_tmps_cone(llwork, rr, op, Id, ts, b, truncp)
        push!(gen_r2sR, ents)


        overlapLR = overlap_noconj(ll,rr)

        #TODO  renormalize by overlap ?
        ll *= sqrt(1/overlapLR)
        rr *= sqrt(1/overlapLR)

        evs_computed = compute_expvals(ll, rr, ["all"], b)
        mergedicts!(expvals, evs_computed)


        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)

        ent = vn_entanglement_entropy(ll)

        if save_cp && length(ll) > 50 && length(ll) % 20 == 0
            jldsave("cp_cone_$(length(ll))_chi_$(chis[end]).jld2"; psi, ll, rr, chis, expvals, entropies, infos)
        end

        push!(vn_ents, ent)
        push!(times, length(ll)*tp.mp.dt)

        next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end


    return ll, rr, chis, expvals, entropies, infos
end


""" Resumes a light cone simulation from a checkpoint file """
function resume_cone(checkpoint::String, nsteps::Int)

    c = jldopen(checkpoint, "r")

    psi = c["psi"]
    op = c["infos"][:op]
    tp = c["infos"][:tp]
    truncp = c["infos"][:truncp]

    # TODO extend with prev results 
    return run_cone(psi, nsteps, op, tp, truncp)
    
end


