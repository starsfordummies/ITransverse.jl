using Revise
using LinearAlgebra, ITensors, JLD2, Dates, Plots

using ITransverse
using ITransverse.ITenUtils

using ProgressMeter

ITensors.enable_debug_checks()

function build_lenvs(psi::MPS, phi::MPS)
    lenvs =[] 

    left_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[1:end-1], phi[1:end-1]))
        left_env =  left_env * Ai
        left_env = left_env * prime(Bi, commoninds(Bi,linkinds(phi)))   
        push!(lenvs, left_env)
    end

    return lenvs
end

function build_renvs(psi::MPS, phi::MPS)
    renvs = [] 

    right_env = ITensor(1.)
    for (ii, (Ai,Bi)) in enumerate(zip(psi[end:-1:1], phi[end:-1:1]))
        right_env =  right_env * Ai
        right_env = right_env * prime(Bi, commoninds(Bi,linkinds(phi)))  
        push!(renvs, right_env)
    end
    return renvs
end

function print_norms_envs(psi::MPS, phi::MPS)
    lenvs = build_lenvs(psi, phi)
    renvs = build_renvs(psi, phi)

    for (ii, (Ai,Bi)) in enumerate(zip(lenvs, renvs))
        @info ii, norm(Ai), norm(Bi)
    end
end

function main_debug_pm()

    plus_state = Vector{ComplexF64}([1/sqrt(2),1/sqrt(2)])

    JXX = 1.0  
    hz = 0.4
    gx = 0.0
    dt = 0.1

    nbeta=0

    init_state = plus_state

    mp = model_params("S=1/2", JXX, hz, gx, dt)
    tp = tmpo_params("S=1/2", "S=1/2", build_expH_ising_parallel_field_murg, mp, nbeta, init_state)

    SVD_cutoff = 1e-20
    maxbondim = 100

    itermax = 200

    verbose=false
    ds2_converged=1e-6

    pm_params = ppm_params(itermax, SVD_cutoff, maxbondim, verbose, ds2_converged)

    sigX = ComplexF64[0,1,1,0]
    sigZ = ComplexF64[1,0,0,-1]
    Id = ComplexF64[1,0,0,1]

    evs = []
    evs2 = []

    Nsteps = 60 
 
    time_sites = siteinds("S=3/2", Nsteps)

    init_mps = build_folded_left_tMPS(tp, time_sites)
    mpo_X = build_folded_tMPO(tp, sigX, time_sites)
    mpo_Z = build_folded_tMPO(tp, sigZ, time_sites)
    mpo_I = build_folded_tMPO(tp, Id, time_sites)


    # init_mps = ITransverse.build_ising_folded_tMPS(build_expH_ising_murg, params, time_sites)

    # mpo_X = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, sigX, time_sites)
    # mpo_Z = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, sigZ, time_sites)
    # mpo_1 = ITransverse.build_ising_folded_tMPO(build_expH_ising_murg, params, Id, time_sites)

    ll, rr, lO, Or, dSs, ents_during_pm = pm_debug(init_mps, mpo_I, mpo_Z, pm_params) # kwargs)

    #ll, rr, lO, Or = pm_debug(init_mps, mpo_1, mpo_X, pm_params) # kwargs)

    @info "Checking (l|Or)"
    ITransverse.check_gencan_left_phipsi(ll,Or)

    @info "Checking (lO|r)"
    ITransverse.check_gencan_left_phipsi(lO,rr)

    #@show ll.llim, ll.rlim, lO.llim, Or.llim

    L1R = overlap_noconj(ll, applys(mpo_I, rr))
    LOR = overlap_noconj(ll, applys(mpo_X, rr))

    ev = LOR/L1R

    L11R = overlap_noconj(apply(swapprime(mpo_1, 0, 1, "Site"), ll, alg="naive",truncate=false), apply(mpo_I, rr,  alg="naive",truncate=false))
    L1OR = overlap_noconj(apply(swapprime(mpo_1, 0, 1, "Site"), ll, alg="naive",truncate=false), apply(mpo_X, rr, alg="naive", truncate=false))
    
    L111R = overlap_noconj(apply(swapprime(mpo_1, 0, 1, "Site"), ll, alg="naive",truncate=false), 
                           apply(mpo_1, apply(mpo_I, rr,  alg="naive",truncate=false),  alg="naive",truncate=false))

    L1O1R = overlap_noconj(apply(swapprime(mpo_I, 0, 1, "Site"), ll, alg="naive",truncate=false),
                          apply(mpo_I, apply(mpo_X, rr, alg="naive", truncate=false),  alg="naive",truncate=false))

    ev2 = L1OR/L11R

    ev3 = L1O1R/L111R

    push!(evs, ev)
    push!(evs2, ev2)
    push!(evs2, ev3)

    return evs, evs2,  dSs, ents_during_pm
end



function pm_debug(in_mps::MPS, in_mpo_1::MPO, in_mpo_X::MPO, pm_params::ppm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)


    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    Ol = MPS()
    Or = MPS()

    ents_during_pm = [] 
    dSs = []

    sjj_prev = ones(length(in_mps)-1)

    for jj = 1:itermax  

        # if jj > 5
        #     cutoff = pm_params.cutoff^2
        # else
        #     cutoff = pm_params.cutoff 
        # end

        # Enforce that the overlap is zero in the end 
        #ll_work = normbyfactor(ll, sqrt(overlap_noconj(ll,rr)))
        #rr_work = normbyfactor(rr, sqrt(overlap_noconj(ll,rr)))

        ll_work = deepcopy(ll)
        rr_work = deepcopy(rr)

        @show overlap_noconj(ll_work,rr_work)
 

        OpsiL = apply(in_mpo_1, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_X, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        # OpsiL = apply(in_mpo_1, ll_work, cutoff=1e-20)
        # OpsiR = apply(swapprime(in_mpo_X, 0, 1, "Site"), rr_work,  cutoff=1e-20)  

        # OpsiL = apply(in_mpo_1, OpsiL,  alg="naive", truncate=false)
        # OpsiL = apply(in_mpo_1, OpsiL,  alg="naive", truncate=false)
        # OpsiR = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  
        # OpsiR = apply(swapprime(in_mpo_1, 0, 1, "Site"), OpsiR,  alg="naive", truncate=false)  
        # OpsiR = apply(swapprime(in_mpo_X, 0, 1, "Site"), OpsiR,  alg="naive", truncate=false)  


        ll, Or, sjj, overlap = debug_trunc_sweep(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        push!(ents_during_pm, sjj)
        dS = norm(sjj - sjj_prev)
        push!(dSs, dS)
        @show dS
        sjj_prev = sjj 

        # @error "Before trunc"
        # print_norms_envs(OpsiL,OpsiR)
    
        # @error "after trunc"
        # print_norms_envs(ll,Or)

        # sleep(5)
    
        OpsiL = apply(in_mpo_X, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        Ol, rr, _, overlap = debug_trunc_sweep(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        # overl = overlap_noconj(ll,rr)
        @show inner(normalize(ll),normalize(ll_work))
        @show inner(normalize(rr),normalize(rr_work))
        # @show overlap_noconj(ll,rr_work)
        # @show overlap_noconj(ll_work,rr)
        # @show norm(ll_work)
        # @show inner(ll_work,ll)

    end

    return ll, rr, Ol, Or, dSs, ents_during_pm

end





""" Debug truncation sweep where we try different things """
function debug_trunc_sweep(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    L_ortho = deepcopy(left_mps)
    R_ortho = deepcopy(right_mps)

    L_ortho = orthogonalize(L_ortho, 3)
    R_ortho = orthogonalize(R_ortho, 3)

    L_ortho = orthogonalize(L_ortho, 1)
    R_ortho = orthogonalize(R_ortho, 1)

    L_ortho = normalize(L_ortho)
    R_ortho = normalize(R_ortho)

    # L_ortho[1] /= norm(L_ortho)
    # R_ortho[1] /= norm(R_ortho)

    #@info "chi Before sweep: $(linkdims(L_ortho))"
    #@info "overlap before everything $(overlap_noconj(left_mps, right_mps))"
    #@info "overlap before trunc $(overlap_noconj(L_ortho, R_ortho))"

    XUinv, XVinv, deltaS = (ITensor(1.), ITensor(1.), ITensor(1.)) 
    left_prev = ITensor(1.)

    ents_sites = Vector{ComplexF64}()

    sweep_factor = 1. 
    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        left_env = left_prev
        left_env *= Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        if norm(left_env) > 10 || norm(left_env) < 0.1 
            @warn "Watch norm env @site $(ii):  $(norm(left_env))"
        end

        norm_left = norm(left_env)
        left_env /= norm_left

        sweep_factor *= 1/sqrt(norm_left)

        U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)

        #@show norm(S)
        #@info "smallest S kept is $(S[end])"
        
        sqS = sqrt.(S)
        isqS = sqS.^(-1)

        XU = dag(U) * isqS 
        XUinv = sqS * U

        XV = dag(Vdag) * isqS
        XVinv = sqS * Vdag

        L_ortho[ii] = Ai * XU / sqrt(norm_left)
        R_ortho[ii] = Bi * XV / sqrt(norm_left)

        left_prev *= L_ortho[ii]
        left_prev *= R_ortho[ii]

        deltaS = delta(inds(S))

        push!(ents_sites, sum(S*log.(S)))
      
    end

    # the last two 
    L_ortho[end] = XUinv * L_ortho[end]
    R_ortho[end] = XVinv * R_ortho[end]

    gen_overlap = scalar(deltaS * ( L_ortho[end] *  R_ortho[end] ))

    #@info "factor: $(sweep_factor) chi after = $(linkdims(L_ortho))"
   
    @info "overlap after trunc $(overlap_noconj(L_ortho, R_ortho)) | $(gen_overlap)"


    return L_ortho, R_ortho, ents_sites, gen_overlap

end






function pm_debug_svd(in_mps::MPS, in_mpo_1::MPO, in_mpo_X::MPO, pm_params::ppm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)


    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    Ol = MPS()
    Or = MPS()

    for jj = 1:itermax  

        # Enforce that the overlap is zero in the end 
        ll_work = normbyfactor(ll, sqrt(overlap_noconj(ll,rr)))
        rr_work = normbyfactor(rr, sqrt(overlap_noconj(ll,rr)))
        @show overlap_noconj(ll_work,rr_work)
 


        ll = apply(in_mpo_1, ll_work, cutoff=1e-12, maxdim=50)
        rr = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr_work, cutoff=1e-12, maxdim=50)  

        @show maxlinkdim(ll), maxlinkdim(rr)
        #overl = overlap_noconj(ll,rr)
        @show overlap_noconj(ll,rr)
        @show overlap_noconj(ll,rr_work)
        @show overlap_noconj(ll_work,rr)
        @show norm(ll_work)
        @show inner(ll_work,ll)

    end

    return ll, rr

end




evs, evs2,  dSs, ents_during_pm = main_debug_pm()
println(evs)
println(ITransverse.ITenUtils.bench_X_04_plus[59])

