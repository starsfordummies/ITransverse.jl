function build_expH_random_symm_svd_1o(dt::Number)
    # 1) Define the site indices for a three-site system
    
    s = siteinds("S=1/2", 3)
    
    # # 2) Prepare arrays of X, Y, Z operators for each site
    # #    We'll index them as [X, Y, Z] for convenience.
    ops_site1 = [op(s, "X", 1), op(s, "Z", 1), op(s, "Y", 1)]
    ops_site2 = [op(s, "X", 2), op(s, "Z", 2), op(s, "Y", 2)]
    ops_site3 = [op(s, "X", 3), op(s, "Z", 3), op(s, "Y", 3)]
    
   
    # s = siteinds("S=1/2", 3)
    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    # Z1 = op(s, "Z", 1)
    # Z2 = op(s, "Z", 2)
    # Z3 = op(s, "Z", 3)

    # Y1 = op(s, "Y", 1)
    # Y2 = op(s, "Y", 2)
    # Y3 = op(s, "Y", 3)

    c1=1 


    e12 = exp(im * dt* ( c1 * X1*X2 ))
    e23 = exp(im * dt* ( c1 * X2*X3 ))

    b = Vector{Float64}(undef, 3)

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

    l1, r2 = ITenUtils.symm_factorization(e12, inds(op(s, "X", 1)), cutoff=1e-14)
    l2, r3 = ITenUtils.symm_factorization(e23, inds(op(s, "X", 2)), cutoff=1e-14)

    Wl = apply(eH1, l1)
    Wc = apply(eH2, apply(r2, l2))
    Wr = apply(eH3, r3)

    # 9) Return the resulting 3-site MPO
    return MPO([Wl, Wc, Wr])
end

function build_expH_random()
   
    s = siteinds("S=1/2", 3)
    
    h = rand(ComplexF64,4,4) 
    h = h + h'

    u = exp(im*h)  # (ij)-(i'j')
    u = reshape(u, 2,2,2,2)
    u = permutedims(u, (1,3,2,4))
    u = reshape(u, 4,4)
    q,r = qr(u)
    
    q = reshape(Matrix(q), 2,2,4)
    r = reshape(r, 4,2,2)

    link1, link2 = Index.(4, ["link1","link2"])

    Wl = ITensor(q, s[1],s[1]', link1)
    Wc = replaceprime(ITensor(r, s[2],s[2]', link1) * ITensor(q, s[2]',s[2]'', link2), 2 =>1)
    Wr = ITensor(r, link2, s[3], s[3]')
    # 9) Return the resulting 3-site MPO
    return MPO([Wl, Wc, Wr])
end