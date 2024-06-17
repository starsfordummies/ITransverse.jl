
"""
One step of the light cone algorithm: takes left and right tMPS ll, rr,
the time MPO and the operator O
extends the time MPO and the left-right tMPS by optimizing 
1) the overlap (ll1|Orr)  -> save new ll
2) the overlap (llO|1rr)  -> save new rr  (in case non symmetric)

Returns the updated left-right tMPS 
"""
function gpu_extend_tmps_cone(ll::AbstractMPS, rr::AbstractMPS, 
    op_L::Vector{ComplexF64}, op_R::Vector{ComplexF64}, 
    tp::tmpo_params,
    truncp::trunc_params)

    time_sites = siteinds("S=3/2", length(ll)+1)
    time_sites= addtags(time_sites, "time_fold")

    tmpo = NDTensors.cu(build_ham_folded_tMPO(tp, op_L, time_sites))

    #println("check: " , length(tmpo), length(ll), length(rr))

    psi_L = gpu_apply_extend(tmpo, ll)

    tmpo = NDTensors.cu(swapprime(build_ham_folded_tMPO(tp, op_R, time_sites), 0, 1, "Site"))
    psi_R = gpu_apply_extend(tmpo, rr)

    #println(typeof(psi_L[2].tensor.storage))
    #println(typeof(psi_R[2].tensor.storage))
    ll, rr = gpu_truncate_sweep!(psi_L,psi_R, cutoff=truncp.cutoff, chi_max=truncp.maxbondim)
    
    #gen_renyi2 = generalized_renyi_entropy(ll, rr, 2, normalize=true)


    return ll,rr

end


function gpu_apply_extend(A::MPO, ψ::AbstractMPS, close_op::Vector = ComplexF64[1,0,0,0])

    A = sim(linkinds, A)
    ψ = sim(linkinds, ψ)
    
    #@assert length(A) == length(ψ) + 1 

    N = length(ψ)

    ψ_out = MPS(N+1)


    # First site: we close with a [1,0,0,0] (should be ok up to normalization)
    ψ_out[1] = A[1] * NDTensors.cu(ITensor(close_op, siteind(A,1)))

    for j in 1:N
        ψ_out[j+1] = A[j+1] * ψ[j] * delta(siteind(ψ,j), siteind(A,j+1)) # NDTensors.cu(dense(
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




function ITransverse.gpu_run_cone(psi::AbstractMPS, 
    nsteps::Int, 
    op::Vector{ComplexF64}, 
    tp::tmpo_params,
    truncp::trunc_params,
    save_cp::Bool=true
    )

    ll = NDTensors.cu(deepcopy(psi))
    rr = NDTensors.cu(deepcopy(psi))

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
    infos = Dict(:ts => ts, :truncp => truncp, :tp => tp, :op => op)

    p = Progress(nsteps; desc="[GPU cone] $cutoff=$(truncp.cutoff), maxbondim=$(truncp.maxbondim)), method=$(truncp.ortho_method)", showspeed=true) 

    for dt = 1:nsteps
        #println("Evolving $dt")
        llwork = deepcopy(ll)

        # if we're worried about symmetry, evolve separately L and R 
        ll,_ = gpu_extend_tmps_cone(llwork, rr, Id, op, tp, truncp)

        _,rr = gpu_extend_tmps_cone(llwork, rr, op, Id, tp, truncp)

        overlapLR = overlap_noconj(ll,rr)

        #@show overlapLR

        #println("lens: ", length(ll), "     ", length(rr))
        #@show (overlap_noconj(ll,rr))
        #@show maxlinkdim(ll), maxlinkdim(rr)

        #TODO  renormalize by overlap ?
        ll = ll * sqrt(1/overlapLR)
        rr = rr * sqrt(1/overlapLR)


        push!(evs_x, gpu_expval_LR(ll, rr, ComplexF64[0,1,1,0], tp))
        push!(evs_z, gpu_expval_LR(ll, rr, ComplexF64[1,0,0,-1], tp))

        push!(chis, maxlinkdim(ll))
        push!(overlaps, overlapLR)


        if save_cp && length(ll) > 50 && length(ll) % 20 == 0
            llw = NDTensors.cpu(ll)
            rrw = NDTensors.cpu(rr)
            @info "Writing checkpoint file"
            jldsave("cp_cone_$(length(ll))_chi_$(chis[end]).jld2"; llw, rrw, chis, expvals, entropies, infos)
        end
     
        push!(ts, length(ll)*tp.dt)
        next!(p; showvalues = [(:Info,"[$(dt)] χ=$(maxlinkdim(ll)), (L|R) = $overlapLR " )])

    end

    return ll, rr, chis, expvals, entropies, infos

end

