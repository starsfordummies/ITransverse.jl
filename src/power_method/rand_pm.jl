
using Revise
using ITensors, JLD2
using ProgressMeter

using ITransverse

ITensors.enable_debug_checks()

""" Setup for a Loschmidt-type tMPS but we apply different TMPO (possibly chosen randomly) at each timestep """
coin(w = 0.5) = rand() < w;

function pm_rand(in_mps::MPS, mpo0::MPO, mpo1::MPO, pm_params::PMParams)

    (; opt_method, itermax, eps_converged, truncp) = pm_params
    (; cutoff, maxbondim) = truncp

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    sprevs = fill(1., length(in_mps)-1)
    LRprev = overlap_noconj(in_mps,in_mps)

    p = Progress(itermax; desc="[PM|$(opt_method)] L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    info_iterations = Dict(:ds2 => ComplexF64[], :RRnew => ComplexF64[], :LRdiff => ComplexF64[] )

    for jj = 1:itermax  

        # which random MPO to take ?
        mpo_L = coin() ? mpo0 : mpo1
        mpo_R = coin() ? mpo0 : mpo1

        if opt_method == "LR"
        
            rr_work = rr
            ll_work = ll 
    
            # optimize <LO|1R> -> new |R> 
            OpsiR = applyn(mpo_R, rr_work)
            OpsiL = applyns(mpo_L, ll_work)  

            OpsiR = normalize(OpsiR)
            OpsiL = normalize(OpsiL)

            rr, ll, sjj = truncate_rsweep(OpsiR, OpsiL, cutoff=cutoff, chi_max=maxbondim)

        elseif opt_method == "RTE"
            #rr_work = normbyfactor(rr, sqrt(overlap_noconj(rr,rr)))
            rr_work = normalize(rr)

            rr = apply(mpo_R, rr_work, cutoff=cutoff, maxdim=maxbondim)
            # work = normalize(ll)
            # ll = apply(mpo_L, work, cutoff=cutoff, maxdim=maxbondim)

            sjj = vn_entanglement_entropy(rr)

        else
            @error "Wrong optimization method: $opt_method"
        end


        # If I cook them separately, likely the overlap will be messed up 
        LRnew = overlap_noconj(ll,rr)
        push!(info_iterations[:LRdiff], abs(LRnew-LRprev))
        LRprev = LRnew
   
        if abs(LRnew) < 1e-6
            @warn "Small overlap $LRnew, watch for trunc error"
        end

        ds2 = norm(sprevs - sjj)
        push!(info_iterations[:ds2], ds2)
        # push!(ds2s, ds2)
        sprevs = sjj

        RRnew = inner(rr_work,rr)/norm(rr)/norm(rr_work)

        push!(info_iterations[:RRnew], RRnew)

        maxnormS = maximum([norm(ss for ss in sjj)])

        if ds2 < eps_converged
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        next!(p; showvalues = [(:Info,"[$(jj)][χ=$(maxlinkdim(ll))] ds2=$(ds2), <R|Rnew>=1-$(round(1-RRnew,digits=8)) |S|=$(maxnormS)" )])

    end

    return ll, rr, info_iterations

end




function main_fw_randpm()

    JXX = 1.0
    hz = 1.0
    gx = 0.0

    dt = 0.1

    up_state = Vector{ComplexF64}([1, 0])
    down_state = Vector{ComplexF64}([0, 1])

    #plus_state = Vector{ComplexF64}([1 / sqrt(2), 1 / sqrt(2)])
    #init_state = plus_state

    init_state = up_state

    nbeta = 0 

    cutoff = 1e-16
    maxbondim = 140
    itermax = 10
    eps_converged = 1e-6

    truncp = TruncParams(cutoff, maxbondim)

    pm_params = PMParams(truncp, itermax, eps_converged, true, "RTM_LR")

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params(build_expH_ising_murg, mp, nbeta, init_state, init_state)



    b = FwtMPOBlocks(tp)
    mpim = model_params(tp.mp; dt=-im*tp.mp.dt)
    tpim = tmpo_params(tp; mp=mpim)
    b_im = FwtMPOBlocks(tpim)

    ts = 50

    Ntime_steps = ts
    Nsteps = nbeta + Ntime_steps + nbeta

    time_sites = addtags(siteinds("S=1/2", Nsteps; conserve_qns=false), "time")


    mpo_up, start_mps = fw_tMPOn(b, b_im, time_sites, right_state = ITensor(up_state, Index(2)))
    mpo_down, _       = fw_tMPOn(b, b_im, time_sites, right_state = ITensor(down_state, Index(2)))


    lls = Vector{MPS}()
    rrs = Vector{MPS}()
    infos = Dict[]

    for nruns = 1:3
    ll, rr, info = pm_rand(start_mps, mpo_up, mpo_down, pm_params)

    push!(lls, ll)
    push!(rrs, rr)
    push!(infos, info)
    end 


    # if ts % 20 == 0
    #     out_filename = "cp_ising_$(ts)_$(maxlinkdim(psi_trunc)).jld2"
    #     jldsave(out_filename; ll_murgs, ds2s, tp, pm_params, curr_T, allts)
    # end

    return lls, rrs, infos

end
    
ll, rr, infos = main_fw_randpm()
