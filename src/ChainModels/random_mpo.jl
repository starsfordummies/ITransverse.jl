
function build_expH_random_symm_svd_1o(dt::Number)
    # 1) Define the site indices for a three-site system
    
    s = siteinds("S=1/2", 3)
    
    # # 2) Prepare arrays of X, Y, Z operators for each site
    # #    We'll index them as [X, Y, Z] for convenience.
    ops_site1 = [op(s, "X", 1), op(s, "Z", 1), op(s, "Y", 1)]
    ops_site2 = [op(s, "X", 2), op(s, "Z", 2), op(s, "Y", 2)]
    ops_site3 = [op(s, "X", 3), op(s, "Z", 3), op(s, "Y", 3)]
    
   
    # c = zeros(Float64, 3, 3)
    # for α in 1:3
    #     for β in α:3
    #         val = 0.5 + randn()
    #         c[α, β] = val
    #         c[β, α] = val
    #     end
    # end

    # println(is_symmetric(c))

    # # 4) Generate random single-site field components b[γ], γ ∈ {x=1, y=2, z=3},
    # #    with the same values used on sites 1, 2, and 3
    # b = Vector{Float64}(undef, 3)
    # for γ in 1:3
    #     b[γ] =  randn()
    # end


    # # 4) Build the two-site Hamiltonians (same c for each bond)
    # #    O12 = Σ_{α,β} c[α,β] * σ1^α ⊗ σ2^β
    # #    O23 = Σ_{α,β} c[α,β] * σ2^α ⊗ σ3^β
    # O12 = ITensor()
    # for α in 1:2
    #     β=α
    #         term = c[α, β] * (ops_site1[α] * ops_site2[β])
    #         if α == 1 && β == 1
    #             O12 = term
    #         else
    #             O12 += term
    #         end
        
    # end

    # O23 = ITensor()
    # for α in 1:2
    #     β=α
    #         term = c[α, β] * (ops_site2[α] * ops_site3[β])
    #         if α == 1 && β == 1
    #             O23 = term
    #         else
    #             O23 += term
    #         end
        
    # end

    # e12 = exp(im * dt * O12)
    # e23 = exp(im * dt * O23)

    # s = siteinds("S=1/2", 3)
    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    Z1 = op(s, "Z", 1)
    Z2 = op(s, "Z", 2)
    Z3 = op(s, "Z", 3)

    Y1 = op(s, "Y", 1)
    Y2 = op(s, "Y", 2)
    Y3 = op(s, "Y", 3)

    c1=1 


    e12 = exp(im * dt* ( c1 * X1*X2 ))
    e23 = exp(im * dt* ( c1 * X2*X3 ))

 

    # 5) Build single-site Hamiltonians (b is the same across sites)
    #    H1 = Σ_{γ} b[γ] * σ1^γ
    #    H2 = Σ_{γ} b[γ] * σ2^γ
    #    H3 = Σ_{γ} b[γ] * σ3^γ


 # 4) Generate random single-site field components b[γ], γ ∈ {x=1, y=2, z=3},
    #    with the same values used on sites 1, 2, and 3
    b = Vector{Float64}(undef, 3)
    # for γ in 1:3
    #     b[γ] =  randn()
    # end

    b = Vector{Float64}(undef, 3)
    for γ in 1:3
        b[γ] =  randn()/sqrt(2)
    end

    H1 = ITensor()
    for γ in 1:3
        term = b[γ] * ops_site1[γ]
        H1 = (γ == 1) ? term : (H1 + term)
    end

    H2 = ITensor()
    for γ in 1:3
        term = b[γ] * ops_site2[γ]
        H2 = (γ == 1) ? term : (H2 + term)
    end

    H3 = ITensor()
    for γ in 1:3
        term = b[γ] * ops_site3[γ]
        H3 = (γ == 1) ? term : (H3 + term)
    end

   
  
    eH1 = exp(im * dt * H1)
    eH2 = exp(im * dt * H2)
    eH3 = exp(im * dt * H3)

    # 7) Symmetric factorization of e12 and e23
    #    (Using your same factorization approach as in the Ising code)
    l1, r2 = ITenUtils.symm_factorization(e12, inds(op(s, "X", 1)), cutoff=1e-14)
    l2, r3 = ITenUtils.symm_factorization(e23, inds(op(s, "X", 2)), cutoff=1e-14)

    # 8) Combine single-site exponentials and two-site factors
    #    Mirroring the structure of your XX+Z code:
    #    Wl = eH1 * l1
    #    Wc = eH2 * (r2 * l2)
    #    Wr = eH3 * r3
    # We use 'apply' if you prefer that style.
    Wl = apply(eH1, l1)
    Wc = apply(eH2, apply(r2, l2))
    Wr = apply(eH3, r3)

    # 9) Return the resulting 3-site MPO
    return MPO([Wl, Wc, Wr])
end