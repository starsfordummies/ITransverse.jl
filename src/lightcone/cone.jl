""" Initializes the light cone folded and rotated temporal MPS |R> given `tmpo_params`
builds a (length 1) tMPS with a single tensor with one (time,physical) open leg.
After rotation, initial state goes to the *left*."""
function init_cone(tp::tmpo_params)

    eH = tp.expH_func(tp.mp)
    cone_mps = init_cone(eH, tp.bl)

    return cone_mps

end

function init_cone(eH_space::MPO, init_state::Vector{<:Number})
    
    Wr = eH_space[3]

    ilink = linkind(eH_space,2)
    iphys = siteind(eH_space,3)

    #fold 
    WWr = Wr * dag(prime(Wr,2))

    # Combine inds 
    Cv = combiner(ilink,ilink''; tags="cwR")
    # we flip the p<>* legs on the backwards tensors for the folding
    Cp = combiner(iphys',iphys''; tags="cp")


    # (in rotated indices, we trace to the right and contract with the initial state to the left
    WWr *= delta(iphys,iphys''')
    WWr *= Cv 
    WWr *= Cp

    rho0 = init_state * init_state'
    init_tensor = ITensor(rho0, combinedind(Cp))

    WWr = WWr * init_tensor

    tMPS = MPS(1)

    # TODO make this more generic
    time_sites = siteinds(dim(combinedind(Cv)), 1)

    time_sites= addtags(time_sites, "time_fold")

    tMPS[1] = WWr * delta(combinedind(Cv), time_sites[1]) 

    return tMPS

end



"""
One step of the light cone algorithm: takes left and right tMPS ll, rr,
the time MPO and the operator O
extends the time MPO and the left-right tMPS by optimizing 
1) the overlap (ll1|Orr)  -> save new ll
2) the overlap (llO|1rr)  -> save new rr  (in case non symmetric)

Returns the updated left-right tMPS 
"""
function extend_tmps_cone(ll::MPS, rr::MPS, 
    op_L::Vector{<:Number}, op_R::Vector{<:Number}, 
    tp::tmpo_params,
    truncp::trunc_params,
    compute_r2::Bool=false)

    time_sites = siteinds(dim(siteind(ll,1)), length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    tmpo = folded_tMPO(tp, time_sites; fold_op=op_L)

    #println("check: " , length(tmpo), length(ll), length(rr))

    psi_L = apply_extend(tmpo, ll)

    tmpo = swapprime(folded_tMPO(tp, time_sites; fold_op=op_R), 0, 1, "Site")
    psi_R = apply_extend(tmpo, rr)

    # ! CHECK CAN WE EVER HAVE LINKS (L) != LINKS (R) ??? WHY DOES EIGEN FAIL??
    ll, rr, ents = truncate_normalize_sweep(psi_L,psi_R, truncp)
    
    gen_renyi2 = [0.]
    if compute_r2
        gen_renyi2 = generalized_renyi_entropy(ll, rr, 2, normalize=true)
    end

    return ll,rr, gen_renyi2 # ents

end


""" Extends MPS ψ to the *left* by one site by applying the MPO A on top,
Returns a new MPS which is the extension of ψ (extended to the left), with siteinds matching those of A
The p' leg of the MPO on the first site is closed by a `close_op` vector 
```
      | | | | | | |
 (op)-o-o-o-o-o-o-o-(in)
      v | | | | | |
        o-o-o-o-o-o
``` 
"""
function apply_extend(A::MPO, ψ::MPS, close_op::Vector = ComplexF64[1,0,0,0])

    A = sim(linkinds, A)
    ψ = sim(linkinds, ψ)
    
    @assert length(A) == length(ψ) + 1 

    N = length(ψ)

    ψ_out = MPS(N+1)

    # First site: we close with a [1,0,0,0] (should be ok up to normalization)
    ψ_out[1] = A[1] * ITensor(close_op, siteind(A,1))

    for j in 1:N
        ψ_out[j+1] = A[j+1] * ψ[j] * delta(siteind(ψ,j), siteind(A,j+1))
    end
    
    # fix links
    for b in 1:N
        Al = commoninds(A[b], A[b + 1])
        ψl = []
        if b > 1 
            ψl = commoninds(ψ[b-1], ψ[b])
        end
        l = [Al..., ψl...]
        #println(b, l)
        if !isempty(l)
        C = combiner(l)
        ψ_out[b] *= C
        ψ_out[b + 1] *= dag(C)
        end
    end

    return noprime(ψ_out)

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
    ts = [] 

    entropies = Dict(:genr2L => gen_r2sL, :genr2R => gen_r2sR, :vn => vn_ents)

    which_evs = ["X","Z","eps"]
    expvals = Dict()
    for op in which_evs
        expvals[op] = []
    end


    infos = Dict(:ts => ts, :truncp => truncp, :tp => tp, :op => op)


    p = Progress(nsteps; desc="[cone] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim)), method=$(truncp.ortho_method)", showspeed=true) 

    for dt = 1:nsteps
        
        # Original should work 
        llwork = deepcopy(ll)
        # if we're worried about symmetry, evolve separately L and R 
        ll,_, ents = extend_tmps_cone(llwork, rr, Id, op, tp, truncp)
        push!(gen_r2sL, ents)

        _,rr, ents = extend_tmps_cone(llwork, rr, op, Id, tp, truncp)
        push!(gen_r2sR, ents)

        #= 
        # Alternative: 
        # if we're worried about symmetry, evolve separately L and R 
        llwork = deepcopy(ll)

        ll,_, ents = extend_tmps_cone_alt(llwork, rr, Id, op, tp, truncp)
        push!(gen_r2sL, ents)

        _,rr, ents = extend_tmps_cone_alt(llwork, rr, op, Id, tp, truncp)
        push!(gen_r2sR, ents)
        =# 


        overlapLR = overlap_noconj(ll,rr)

        #TODO  renormalize by overlap ?
        ll = ll * sqrt(1/overlapLR)
        rr = rr * sqrt(1/overlapLR)

        evs_computed = compute_expvals(ll, rr, ["all"], tp)
        mergedicts!(expvals, evs_computed)


        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)

        llc = deepcopy(ll)
        orthogonalize!(llc,1)
        ent = vn_entanglement_entropy(llc)

        if save_cp && length(ll) > 50 && length(ll) % 20 == 0
            jldsave("cp_cone_$(length(ll))_chi_$(chis[end]).jld2"; psi, ll, rr, chis, expvals, entropies, infos)
        end

        push!(vn_ents, ent)
        push!(ts, length(ll)*tp.mp.dt)

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

function extend_tmps_cone_alt(ll_in::MPS, rr_in::MPS, 
    op_L::Vector{ComplexF64}, op_R::Vector{ComplexF64}, 
    tp::tmpo_params,
    truncp::trunc_params,
    compute_r2::Bool=false)

    ll = deepcopy(ll_in)
    rr = deepcopy(rr_in)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    tmpo = build_folded_tMPO(tp, op_L, time_sites)

    psin = ITensor(ComplexF64[1,0,0,0], siteind(tmpo,1))
    insert!(ll.data, 1, psin)
    replace_siteinds!(ll, siteinds(tmpo))
    psi_L = apply(tmpo, ll, alg="naive", truncate=false)

    tmpo = swapprime(build_folded_tMPO(tp, op_R, time_sites), 0, 1, "Site")
    psin = ITensor(ComplexF64[1,0,0,0], siteind(tmpo,1))
    insert!(rr.data, 1, psin)
    replace_siteinds!(rr, siteinds(tmpo))
    psi_R = apply(tmpo, rr)

    # ! CHECK CAN WE EVER HAVE LINKS (L) != LINKS (R) ??? WHY DOES EIGEN FAIL??
    ll, rr, ents = truncate_normalize_sweep(psi_L,psi_R, truncp)
    
    gen_renyi2 = [0.]
    if compute_r2
        gen_renyi2 = generalized_renyi_entropy(ll, rr, 2, normalize=true)
    end

    return ll,rr, gen_renyi2 # ents

end

