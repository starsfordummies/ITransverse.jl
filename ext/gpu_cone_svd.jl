function gpu_extend_tmps_cone_sym_svd(ll::AbstractMPS, 
    op_L::Vector{ComplexF64},
    tp::tmpo_params,
    truncp::trunc_params)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    tmpo = NDTensors.cu(build_ham_folded_tMPO(tp, op_L, time_sites))

    psin = NDTensors.cu(ITensor(ComplexF64[1,0,0,0], siteind(tmpo,1)))
    insert!(ll.data, 1, psin)

    replace_siteinds!(ll, time_sites)
    apply(tmpo, ll, cutoff=truncp.cutoff)

    return ll

end




function gpu_expval_cone_sym(ll::MPS, op::Vector{ComplexF64}, tp::tmpo_params)

    fold_id = ComplexF64[1,0,0,1]

    time_sites = siteinds(ll)
    tmpo = NDTensors.cu(build_ham_folded_tMPO(tp,  fold_id, time_sites))
    psi_L = apply(tmpo, ll, alg="naive", truncate=false)

    tmpo = NDTensors.cu(swapprime(build_ham_folded_tMPO(tp, op, time_sites), 0, 1, "Site"))
    psi_R = apply(tmpo, ll, alg="naive", truncate=false)

    ev_LOR = overlap_noconj(psi_L,psi_R)
    ev_L1R = overlap_noconj(psi_L,psi_L)

    ev = ev_LOR/ev_L1R

    return ev

end



function ITransverse.gpu_run_cone_svd(psi::MPS, 
    nsteps::Int, 
    tp::tmpo_params,
    truncp::trunc_params,
    save_cp::Bool=true
    )

    ll = deepcopy(psi)

    Id = ComplexF64[1,0,0,1]

    evs_x = []
    evs_z = []
    chis = []
    overlaps = []
    vn_ents = []
    gen_r2sL = []
    gen_r2sR = []
    ts = [] 

    entropies = Dict(:genr2L => gen_r2sL, :genr2R => gen_r2sR, :vn => vn_ents)
    expvals = Dict(:evs_x => evs_x, :evs_z => evs_z, :overlaps => overlaps)
    infos = Dict(:ts => ts, :truncp => truncp, :tp => tp, :op => Id)


    p = Progress(nsteps; desc="[cone] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim)), method=$(truncp.ortho_method)", showspeed=true) 

    for dt = 1:nsteps
        #println("Evolving $dt")
        llwork = NDTensors.cu(deepcopy(ll))

        # if we're worried about symmetry, evolve separately L and R 
        ll = gpu_extend_tmps_cone_sym_svd(llwork, Id, tp, truncp)

        overlapLR = overlap_noconj(ll,ll)

        #TODO  renormalize by overlap ?
        ll = ll * sqrt(1/overlapLR)

        push!(evs_x, gpu_expval_cone_sym(ll, ComplexF64[0,1,1,0], tp))
        push!(evs_z, gpu_expval_cone_sym(ll, ComplexF64[1,0,0,-1], tp))

        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)

        # llc = deepcopy(ll)
        # orthogonalize!(llc,1)
        # ent = vn_entanglement_entropy(llc)
        ent= [0.] # TODO 

        if save_cp && length(ll) > 50 && length(ll) % 20 == 0
            jldsave("cp_cone_$(length(ll))_chi_$(chis[end])_sym.jld2"; psi, ll, chis, expvals, entropies, infos)
        end

        push!(vn_ents, ent)
        push!(ts, length(ll)*tp.dt)

        next!(p; showvalues = [(:Info,"[$(length(ll))] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end


    return ll, ll, chis, expvals, entropies, infos
end
