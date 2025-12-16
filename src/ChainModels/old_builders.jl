function build_expH_ising_murg_old(
    sites::Vector{<:Index},
    JXX::Real,
    gz::Real,
    λx::Real)

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2

    #link_indices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]
    link_indices = hasqns(sites) ?
        [Index([QN("SzParity", 1, 2) => 1, QN("SzParity", 0, 2) => 1], "Link,l=$(n-1)") for n = 1:N+1] : 
        [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]


    for n = 1:N
        # siteindex s

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(link_indices[n])
        # right link index rl
        rl = link_indices[n+1]

        I = op(sites, "Id", n) 
        X = op(sites, "X", n)

        if n == 1
            #U_t[n] = ITensor(ComplexF64, dag(s), s', dag(rl))
            U_t[n] = onehot(rl => 1) * sqrt(cos(JXX))*I
            U_t[n] += onehot(rl => 2) * sqrt(im*sin(JXX))*X
        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] = onehot(ll => 1) * sqrt(cos(JXX))*I
            U_t[n] += onehot(ll => 2) * sqrt(im*sin(JXX))*X

        else
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_t[n] = onehot(ll => 1, rl =>1) * cos(JXX)*I
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(JXX))*sqrt(cos(JXX))*X
            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(JXX))*sqrt(cos(JXX))*X
            U_t[n] += onehot(ll => 2, rl =>2) * im*sin(JXX)*I
        end

        Ux = exp(im*λx*op(sites, "X", n))
        Uz2 = exp(0.5*im*gz*op(sites, "Z", n))


        # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
        # everything is symmetric in phys legs here so no need to worry too much
        # (otherwise this is not right, transpositions!) 
        # TODO CHECK ORDER
        U_t[n] *= Uz2' 
        U_t[n] *= Ux
        U_t[n] *= Uz2
        U_t[n] = replaceprime(U_t[n], 2 => 1)

    end

    return U_t


end





function build_murg_bulk_tensor(JXX::Real, gz::Real, λx::Real; phys_ind::Index = siteind("S=1/2"), link_inds=Index.([2,2]))

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    (iL, iR) = link_inds
    
    X = op(phys_ind, "X")
    Z = op(phys_ind, "Z")
    I = op(phys_ind, "I")

    U_XX = onehot(iL => 1, iR =>1) * cos(JXX)*I 
    U_XX += onehot(iL => 1, iR =>2) * sqrt(im*sin(JXX))*sqrt(cos(JXX))*X
    U_XX += onehot(iL => 2, iR =>1) * sqrt(im*sin(JXX))*sqrt(cos(JXX))*X
    U_XX += onehot(iL => 2, iR =>2) * im*sin(JXX)*I

    eX = exp(im*λx*X)
    eZ2 = exp(0.5*im*gz*Z)
    

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    W = apply(U_XX, eZ2) 
    W = apply(eX, W) 
    W = apply(eZ2, W) 

    return W
end


function build_murg_edge_tensor(JXX::Real, gz::Real, λx::Real; phys_ind::Index = siteind("S=1/2"), link_ind=Index(2))

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    X = op(phys_ind, "X")
    Z = op(phys_ind, "Z")
    I = op(phys_ind, "I")

    U_XX = onehot(link_ind => 1) * sqrt(cos(JXX))*I
    U_XX += onehot(link_ind => 2) * sqrt(im*sin(JXX))*X

    eX = exp(im*λx*X)
    eZ2 = exp(0.5*im*gz*Z)
    
    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    W = apply(U_XX, eZ2) 
    W = apply(eX, W) 
    W = apply(eZ2, W) 

    return W
end

function build_expH_ising_murg_new_from_blocks(
    sites::Vector{<:Index},
    JXX::Real,
    gz::Real,
    λx::Real,
    dt::Number)

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    JXX = JXX*dt
    gz = gz*dt
    λx = λx*dt

    N = length(sites)
    
    link_dimension = 2
    link_indices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N-1]

    mpo_tensors = Array{ITensor}(undef, N)

    mpo_tensors[1] = build_murg_edge_tensor(JXX, gz, λx; phys_ind = sites[1], link_ind = dag(link_indices[1]))
    for ii = 2:N-1
        mpo_tensors[ii] = build_murg_bulk_tensor(JXX, gz, λx; phys_ind = sites[ii], link_inds=(link_indices[ii-1], dag(link_indices[ii])))
    end
    mpo_tensors[end] = build_murg_edge_tensor(JXX, gz, λx; phys_ind = sites[end], link_ind = link_indices[end])
    
    U_t = MPO(mpo_tensors)

    return U_t
end



function build_expH_ising_symm_svd_1o(p::IsingParams, dt::Number)
    
    JXX = p.Jtwo*dt
    hz = p.gpar*dt

    s = siteinds("S=1/2", 3)
    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    e12 = exp(im*X1*X2*JXX)
    e23 = exp(im*X2*X3*JXX)

    eZ1 = exp(im*hz*op(s,"Z",1))
    eZ2 = exp(im*hz*op(s,"Z",2))
    eZ3 = exp(im*hz*op(s,"Z",3))

    l1, r2 = ITenUtils.symm_factorization(e12, inds(X1))
    l2, r3 = ITenUtils.symm_factorization(e23, inds(X2))

    # apply(r2,l2) ≈ apply(l2,r2)   #true

    Wl = apply(eZ1, l1)
    Wc = apply(eZ2, apply(r2,l2))
    Wr = apply(eZ3, r3)

    return MPO([Wl, Wc, Wr])

end



""" Potts H MPO built manually with lower-triangular form """
function build_H_potts_manual_lowtri(sites_potts, JJ::Real, ff::Real)
 
    N = length(sites_potts)

    link_dimension = 4

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_x = MPO(N)


    for n = 1:N

        ll = dag(linkindices[n])

        # right link index rl - labels the columns
        rl = linkindices[n+1]

        I = op(sites_potts, "Id", n)

        Σ = op(sites_potts, "Σ", n)
        Σd = op(sites_potts, "Σdag", n)
        ττd = op(sites_potts, "τplusτdag",  n)


        # Lower tri
        # Init ITensor inside MPO
        if n == 1
            # Row vector at the left 
            U_x[n] =  onehot(rl => 1) * -ff*ττd
            U_x[n] += onehot(rl => 2) * -JJ*Σd
            U_x[n] += onehot(rl => 3) * -JJ*Σ
            U_x[n] += onehot(rl => 4) * I


        elseif n == N
            #U_x[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_x[n] =  onehot(ll => 1) * I
            U_x[n] += onehot(ll => 2) * Σ
            U_x[n] += onehot(ll => 3) * Σd
            U_x[n] += onehot(ll => 4) * -ff*ττd

        else

            U_x[n] =  onehot(ll => 1, rl =>1) * I
            U_x[n] += onehot(ll => 2, rl =>1) * Σ
            U_x[n] += onehot(ll => 3, rl =>1) * Σd
            U_x[n] += onehot(ll => 4, rl =>1) * -ff*ττd


            U_x[n] += onehot(ll => 4, rl =>2) * -JJ*Σd
            U_x[n] += onehot(ll => 4, rl =>3) * -JJ*Σ
            U_x[n] += onehot(ll => 4, rl =>4) * I

        end

    end

    return U_x
end

