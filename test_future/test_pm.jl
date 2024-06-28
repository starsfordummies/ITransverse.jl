

""" Latest truncate()  """
function truncate_new(left_mps::MPS, right_mps::MPS; method::String, cutoff::Real, chi_max::Int)

    mpslen = length(left_mps)

    L_ortho = orthogonalize(left_mps,  1)
    R_ortho = orthogonalize(right_mps, 1)

    XUinv, XVinv, deltaS = ITensor(1.), ITensor(1.), ITensor(1.)
    #left_prev = ITensor(1.)

    ents_sites = Vector{ComplexF64}()

    # Left gen.can. sweep with truncation 
    for ii = 1:mpslen-1
        Ai = XUinv * L_ortho[ii]
        Bi = XVinv * R_ortho[ii] 

        # Generalized canonical - no complex conjugation!
        left_env = deltaS
        left_env *= Ai 
        left_env *= Bi 

        @assert order(left_env) == 2

        if method == "SVD" 

            # Normalize so sum(sv^2) = 1 
            lnorm = norm(left_env)
            left_env /= lnorm
            
            U,S,Vdag = svd(left_env, ind(left_env,1); cutoff, maxdim=chi_max)

            @assert sum(S.^2) ≈ 1.

            sqS = sqrt.(S)
            isqS = sqS.^(-1)

            XU = dag(U) * isqS / sqrt(lnorm)
            XUinv = sqS * U

            XV = dag(Vdag) * isqS / sqrt(lnorm)
            XVinv = sqS * Vdag


        else
            throw(ArgumentError("need to specify method EIG|SVD - current method=$method"))
        end


        L_ortho[ii] = Ai * XU  
        R_ortho[ii] = Bi * XV

        #left_prev *= L_ortho[ii]
        #left_prev *= R_ortho[ii]
        #left_prev *= deltaS

        # ratnorm = norm(L_ortho[ii])/norm(R_ortho[ii])
        # L_ortho[ii] /= sqrt(lnorm*ratnorm)
        # R_ortho[ii] /= sqrt(lnorm/ratnorm)


        deltaS = delta(inds(S))


        #  if ii > 2 
        # #     #@show inds(S)
        # #     #@info "$ii $(norm(left_prev - deltaS)) $(norm(S))"
        #     if norm(S) > 10000
        #          @show S
        #          @show [norm_gen.(a for a in left_mps)]
        #          @show [norm_gen.(a for a in right_mps)]
        #          sleep(5)
        #     end
        #  end

        #! This could be nasty if we have imaginary stuff...
        #! should not be a problem for SVDs though
        push!(ents_sites, log(sum(S)))
      
    end

    # the last two 
    L_ortho[end] = XUinv * L_ortho[end]
    R_ortho[end] = XVinv * R_ortho[end]

    gen_overlap = deltaS * ( L_ortho[end] *  R_ortho[end] ) 

    return L_ortho, R_ortho, ents_sites, gen_overlap

end


""" Latest PM """
function pm_new(in_mps::MPS, in_mpo_1::MPO, in_mpo_X::MPO, pm_params::pm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim

    converged_ds2 = pm_params.ds2_converged

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    # deltas: renyi2, overlap, eigenvalue

    r2s = Vector{ComplexF64}[]
    ovs = ComplexF64[]
    evs = ComplexF64[]

    d_r2s = Float64[]
    d_ovs = Float64[]
    d_evs = Float64[]

    vals = Dict(:renyi2 => r2s, :overlap => ovs, :eigenvalue => evs)
    deltas = Dict(:renyi2 => d_r2s, :overlap => d_ovs, :eigenvalue => d_evs)

    r2prev = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    lO = MPS()
    Or = MPS()

    for jj = 1:itermax  

        # Enforce that the overlap <L|R> is 1 
        ll_work = normbyfactor(ll, sqrt(overlap_noconj(ll,rr)))
        rr_work = normbyfactor(rr, sqrt(overlap_noconj(ll,rr)))
        #@show overlap_noconj(ll_work,rr_work)
 
        OpsiL = apply(in_mpo_1, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_X, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        ll, Or, sjj, overlap = truncate_new(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        OpsiL = apply(in_mpo_X, ll_work,  alg="naive", truncate=false)
        OpsiR = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        
        lO, rr, _, overlap = truncate_new(OpsiL, OpsiR, cutoff=cutoff, chi_max=maxbondim, method="SVD")

        overl = overlap_noconj(ll,rr)
   
        if abs(overl) < 0.01
            @warn "Small overlap $overl, watch for trunc error"
        end

    


        #TODO 
        r2 = generalized_renyi_entropy(ll,rr,2, normalize=true)

        d_r2 = norm(r2 - r2prev)
        r2prev = r2

        push!(r2s, r2)
   

        if d_r2 < converged_ds2
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        next!(p; showvalues = [(:Info,"[L=$(length(in_mps))][i=$(jj)][χ=$(maxlinkdim(ll))] d_r2=$(d_r2)]")])

    end

    return ll, rr, lO, Or, vals, deltas

end


""" Latest PM """
function pm_new_svd(in_mps::MPS, in_mpo_1::MPO, in_mpo_X::MPO, pm_params::pm_params)

    itermax = pm_params.itermax
    cutoff = pm_params.cutoff
    maxbondim = pm_params.maxbondim

    converged_ds2 = pm_params.ds2_converged

    ll = deepcopy(in_mps)
    rr = deepcopy(in_mps)

    # deltas: renyi2, overlap, eigenvalue

    r2s = Vector{ComplexF64}[]
    ovs = ComplexF64[]
    evs = ComplexF64[]

    d_r2s = Float64[]
    d_ovs = Float64[]
    d_evs = Float64[]

    vals = Dict(:renyi2 => r2s, :overlap => ovs, :eigenvalue => evs)
    deltas = Dict(:renyi2 => d_r2s, :overlap => d_ovs, :eigenvalue => d_evs)

    r2prev = fill(1., length(in_mps)-1)

    p = Progress(itermax; desc="L=$(length(ll)), cutoff=$(cutoff), maxbondim=$(maxbondim))", showspeed=true) 

    lO = MPS()
    Or = MPS()

    for jj = 1:itermax  

        # Enforce that the overlap <L|R> is 1 
        #ll_work = normbyfactor(ll, sqrt(overlap_noconj(ll,rr)))
        #rr_work = normbyfactor(rr, sqrt(overlap_noconj(ll,rr)))
        #@show overlap_noconj(ll_work,rr_work)
 
        ll = apply(in_mpo_1, ll_work,  alg="naive", truncate=false)
        Or = apply(swapprime(in_mpo_X, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        sjj = vn_entanglement_entropy(ll)

        lO = apply(in_mpo_X, ll_work,  alg="naive", truncate=false)
        rr = apply(swapprime(in_mpo_1, 0, 1, "Site"), rr_work,  alg="naive", truncate=false)  

        overl = overlap_noconj(ll,rr)
   
        if abs(overl) < 0.01
            @warn "Small overlap $overl, watch for trunc error"
        end

    
        # #TODO 
        # r2 = generalized_renyi_entropy(ll,rr,2, normalize=true)

        r2 = sjj
        d_r2 = norm(r2 - r2prev)
        r2prev = r2

        push!(r2s, r2)
   

        if d_r2 < converged_ds2
            @info ("[$(length(ll))] converged after $jj steps - χ=$(maxlinkdim(ll))")
            break
        end

        if jj == itermax
            @warn ("NOT converged after $jj steps - χ=$(maxlinkdim(ll))")
        end

        next!(p; showvalues = [(:Info,"[L=$(length(in_mps))][i=$(jj)][χ=$(maxlinkdim(ll))] d_r2=$(d_r2)]")])

    end

    return ll, rr, lO, Or, vals, deltas

end

