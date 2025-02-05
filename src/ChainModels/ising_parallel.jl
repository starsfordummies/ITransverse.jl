function build_H_ising(sites::Vector{<:Index}, mp::IsingParams)
    build_H_ising(sites, mp.Jtwo, mp.gperp, mp.hpar)
end

""" Builds Ising Hamiltonian MPO  H = -Jtwo*XX - gperp*Z - hpar*X """ 
function build_H_ising(sites::Vector{<:Index}, Jtwo::Real, gperp::Real, hpar::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -Jtwo, "X", j, "X", j + 1
    end

    for j in 1:N
        os += -gperp, "Z", j
    end

    for j in 1:N
        os += -hpar, "X", j
    end

    return MPO(os, sites)
end


""" Prescription a la Murg for exp(-i*H*dt) Ising transverse+parallel
Convention H = -( JXX + gzZ + λxX )
"""
function build_expH_ising_murg(
    sites::Vector{<:Index},
    JXX::Real,
    gz::Real,
    λx::Real,
    dt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    # TODO CHECK that this doesn't break anything: 
    JXX = JXX*dt
    gz = gz*dt
    λx = λx*dt

    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2

    link_indices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

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



function build_expH_ising_murg_new(sites::Vector{<:Index},
    JXX::Real, gz::Real, λx::Real, dt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    # TODO CHECK that this doesn't break anything: 
    JXX = JXX*dt
    gz = gz*dt
    λx = λx*dt

    Uxx = build_expXX_murg(sites, JXX)

    sigma_X = [0 1 ; 1 0]
    sigma_Z = [1 0 ; 0 -1]

    eX = exp(im*λx*sigma_X)
    eZ2 = exp(0.5*im*gz*sigma_Z)
    
    Ux = MPO([op(eX, s) for s in siteinds(Uxx, plev=0)])
    Uz2 = MPO([op(eZ2, s) for s in siteinds(Uxx, plev=0)])

    # Ux = MPO(ComplexF64, sites, n -> eX)
    # Uz2 = MPO(ComplexF64, sites, n -> eZ2)

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = applyn(Ux, Uz2) #apply( alg="naive")
    @show U_t[1].tensor

    U_t = applyn(Uxx, U_t)
    @show U_t[1].tensor

    U_t = applyn(Uz2, U_t)
    @show U_t[1].tensor

    @show Uxx[1].tensor
    @show U_t[1].tensor

    @show Uz2[1].tensor

    @show JXX

    return U_t


end

function build_expH_ising_murg_new(s::Vector{<:Index}, p::IsingParams, dt::Number)
    build_expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end


function build_expH_ising_murg_new(p::IsingParams, dt::Number)
    s = siteinds("S=1/2", 3)
    build_expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end



function build_expXX_murg(
    sites::Vector{<:Index},
    Jdt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    N = length(sites)
    U_XX = MPO(N)

    link_dimension = 2

    link_indices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

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
            U_XX[n] = onehot(rl => 1) * sqrt(cos(Jdt))*I
            U_XX[n] += onehot(rl => 2) * sqrt(im*sin(Jdt))*X
        elseif n == N
            #U_XX[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_XX[n] = onehot(ll => 1) * sqrt(cos(Jdt))*I
            U_XX[n] += onehot(ll => 2) * sqrt(im*sin(Jdt))*X

        else
            #U_XX[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_XX[n]  = onehot(ll => 1, rl =>1) * cos(Jdt)*I
            U_XX[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(Jdt))*sqrt(cos(Jdt))*X
            U_XX[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(Jdt))*sqrt(cos(Jdt))*X
            U_XX[n] += onehot(ll => 2, rl =>2) * im*sin(Jdt)*I
        end

    end

    return U_XX


end


function build_expH_ising_murg(s::Vector{<:Index}, p::IsingParams, dt::Number)
    build_expH_ising_murg(s, p.Jtwo, p.gperp, p.hpar, dt)
end

function build_expH_ising_murg(mp::IsingParams, dt::Number)
    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    build_expH_ising_murg(space_sites, mp.Jtwo, mp.gperp, mp.hpar, dt)

end

function build_expH_ising_symm_svd(s::Vector{<:Index}, p::IsingParams, dt::Number)

    w = build_expH_ising_symm_svd(p, dt)
    wmpo = extend_mpo(s, w)
    return wmpo

end


""" Builds core MPO tensors for 3 sites """ 
function build_expH_ising_symm_svd(p::IsingParams, dt::Number)

    s = siteinds("S=1/2", 3)

    JXX = p.Jtwo*dt
    hz = p.gperp*dt
    λx = p.hpar*dt
    
    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    e12 = exp(im*X1*X2*JXX)
    e23 = exp(im*X2*X3*JXX)

    fac_z = hz*0.5
    eZ1 = exp(im*fac_z*op(s,"Z",1))
    eZ2 = exp(im*fac_z*op(s,"Z",2))
    eZ3 = exp(im*fac_z*op(s,"Z",3))

    eX1 = exp(im*λx*op(s,"X",1))
    eX2 = exp(im*λx*op(s,"X",2))
    eX3 = exp(im*λx*op(s,"X",3))

    l1, r2 = ITenUtils.symm_factorization(e12, inds(X1), cutoff=1e-14)
    l2, r3 = ITenUtils.symm_factorization(e23, inds(X2), cutoff=1e-14)

    """

    x   x   x 
    |   |   |
    |   |>=<| 
    |>=<|   |
    |   |   |
    o   o   o
    |   |   | 
    x   x   x 
    L1      R3 
    """
    # apply(r2,l2) ≈ apply(l2,r2)   #true

    Wl = apply(apply(eZ1, apply(eX1, l1)),           eZ1)
    Wc = apply(apply(eZ2, apply(eX2, apply(r2,l2))), eZ2)
    Wr = apply(apply(eZ3, apply(eX3,r3)),            eZ3)

    return MPO([Wl, Wc, Wr])

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




""" Convention XX+Z only for now """
function epsilon_brick_ising(mp::IsingParams)

    temp_s = siteinds("S=1/2",2)
    os = OpSum()
    os += mp.Jtwo,   "X",1,"X",2
    os += mp.gperp/2,  "I",1,"Z",2
    os += mp.gperp/2,  "Z",1,"I",2
    os += mp.hpar/2,  "I",1,"X",2
    os += mp.hpar/2,  "X",1,"I",2

    #ϵ_op = ITensor(os, temp_s, temp_s')
    ϵ_op = MPO(os, temp_s)
    cs1 = combiner(temp_s[1], temp_s[1]')
    cs2 = combiner(temp_s[2], temp_s[2]')
    ϵ_op[1] *= cs1 
    ϵ_op[2] *= cs2 

    return ϵ_op
end
