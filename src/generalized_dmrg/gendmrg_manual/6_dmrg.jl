
# The bulk 

function mygendmrg_manual(Hmpo::MPO, ssites; nsweeps = 4, fact_method = 1)
    
    ψ = randomMPS(ComplexF64, ssites, linkdims=4)
    h_w = Hmpo

    basic_left_can!(ψ, normalize=true)


    left_env = build_left_gen_env(ψ, h_w)
    right_env = build_right_gen_env(ψ, h_w)

    energies_along_sweep = []

    N = length(ψ)

    for isweep = 1:nsweeps

        ######################################################
        # Start from the right, <<<< LEFT SWEEP <<<<<<<
        #####################################################
        for jj = N:-1:2 

            println("[<$(isweep)<] updating $jj (& $(jj-1) )")

            #build Hij 
            Hij = get_renv(right_env,jj+1)

            Hij = Hij * h_w[jj] 
            Hij = Hij * h_w[jj-1]

            Hij = Hij * get_lenv(left_env,jj-2)

            psi0 = ψ[jj] * ψ[jj-1]

            cL = Index(1)
            cR = Index(1)

            if !isempty(linkinds(ψ,jj)) # check we're not at the right edge
                cR = combiner(siteind(ψ,jj), linkind(ψ,jj))
            else
                cR = combiner(siteind(ψ,jj))
            end
            if !isempty(linkinds(ψ,jj-2)) #check we're not at the left edge
                cL = combiner(siteind(ψ,jj-1),linkind(ψ,jj-2)) 
            else
                cL = combiner(siteind(ψ,jj-1))
            end

            #cRp = combiner(siteinds(ψ,jj), linkinds(ψ,jj))' 
            Hij = Hij * cR
            Hij = Hij * cR'
            psi0 = psi0 * cR


            Hij = Hij * cL
            Hij = Hij * cL'
            psi0 = psi0 * cL

            iL = ind(cL,1)
            iR = ind(cR,1)
            # combine prime indices together
            cp0 =  combiner(iL, iR)
            Hij = Hij * cp0
            Hij = Hij * cp0'

            psi0 = psi0 * cp0


            @assert order(Hij) == 2
            @assert order(psi0) == 1

            #Hij = array(Hij)
            #psi0 = array(psi0)

            # Find largest eigenvector of Bij 
            # vals, vecs = eigsolve(
            #     Hij,
            #     psi0,
            #     1,
            #     :LR;
            #     ishermitian=false,
            #     tol=1e-14,
            #     krylovdim=3,
            #     maxiter=1 # ?
            # )
            
            vecs, vals = basic_sym_eig(matrix(Hij))

            energy = vals[1]
            push!(energies_along_sweep, energy)
            phi = ITensor(vecs[:,1], ind(cp0,1))
            phi *= cp0  # or dag(cp0)

            @debug @info " from $(inds(Hij)) to $(inds(phi)))"

            #@show energy, phi
            L, R = factorize_ro(phi, iR, extratags="$(jj-1)", method=fact_method)

            L = L * cL
            R = R * cR 

            #@show inds(L)
            #@show inds(R)
            #@show inds(ψ[jj]), inds(ψ[jj-1])

            ψ[jj] = R
            ψ[jj-1] = L

            update_right_gen_env!(ψ, h_w, right_env, jj)


        end

        @info "Half sweep"
        #@show (inds(ψ))
        check_gen_right_can_form(ψ)

        ################################################
        # >>>>>>>>>> RIGHT SWEEP >>>>>>>>>>>>>>>>>
        ###############################################


        for jj = 1:N-1

            println("[>$(isweep)>] updating $jj (& $(jj+1) )")


            #build Hij 
            Hij = get_lenv(left_env, jj-1)
            Hij = Hij * h_w[jj] 
            Hij = Hij * h_w[jj+1]
            Hij = Hij * get_renv(right_env, jj+2)

            psi0 = ψ[jj] * ψ[jj+1]

            cL = Index(1)
            cR = Index(1)

            @show inds(Hij)

            #TODO need to reshape indices so it's a matrix 
            if !isempty(linkinds(ψ,jj+1)) # check we're not at the right edge
                cR = combiner(siteind(ψ,jj+1), linkind(ψ,jj+1))
            else
                cR = combiner(siteind(ψ,jj+1))
            end
            if !isempty(linkinds(ψ,jj-1)) #check we're not at the left edge
                cL = combiner(siteind(ψ,jj),linkind(ψ,jj-1)) 
            else
                cL = combiner(siteind(ψ,jj))
            end

            Hij = Hij * cR
            Hij = Hij * cR'
            psi0 = psi0 * cR

            Hij = Hij * cL
            Hij = Hij * cL'
            psi0 = psi0 * cL

            iL = ind(cL,1)
            iR = ind(cR,1)
            # combine prime indices together
            cp0 =  combiner(iL, iR)
            Hij = Hij * cp0
            Hij = Hij * cp0'

            psi0 = psi0 * cp0


            @assert order(Hij) == 2
            @assert order(psi0) == 1



            # # Find largest eigenvector of Bij 
            # vals, vecs = eigsolve(
            #     PH,
            #     psi0,
            #     1,
            #     eigsolve_which_eigenvalue=:LR;
            #     ishermitian=false,
            #     tol=1e-14,
            #     krylovdim=3,
            #     maxiter=1 # ?
            # )
            # end


            vecs, vals = basic_sym_eig(matrix(Hij))


            energy = vals[1]
            push!(energies_along_sweep, energy)

            phi = ITensor(vecs[:,1], ind(cp0,1))
            phi *= cp0  # or dag(cp0)

            @debug " from $(inds(Hij)) to $(inds(phi)))"

            #@show energy, phi
            L, R = factorize_lo(phi, iL; extratags="$(jj)", method=fact_method)

            L = L * cL
            R = R * cR 

            @debug @show inds(L)
            @debug @show inds(R)
            @debug @show inds(ψ[jj]), inds(ψ[jj+1])

            ψ[jj] = L
            ψ[jj+1] = R

            update_left_gen_env!(ψ, h_w, left_env, jj)


        end

        @info "End sweep"
        #@show (inds(ψ))
        check_gen_left_can_form(ψ)


    end
    
    @show (inds(ψ))

    return ψ, energies_along_sweep

end

