############ Transverse field Ising ##############
###### Our convention is usually H = -JXX - gZ - hX 
######## Hamiltonian ########

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

######## Time evolution operator exp(-iHt)  ########


""" Symmetric prescription a la Murg for exp(-i*H*dt) Ising transverse+parallel
Convention H = -( JXX + gzZ + λxX )
"""
function build_expH_ising_murg(
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



function build_expH_ising_murg_new(sites::Vector{<:Index},
    JXX::Real, gz::Real, λx::Real, dt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    JXX = JXX*dt
    gz = gz*dt
    λx = λx*dt

    Uxx = build_expXX_murg(sites, JXX)

    Ux = MPO([op(s, "Rx", θ=-2*λx) for s in sites])
    Uz2 = MPO([op(s, "Rz", θ=-gz) for s in sites])


    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λx) ? Uz2 : applyn(Ux, Uz2) 
    U_t = applyn(Uxx, U_t) 
    U_t = applyn(Uz2, U_t) 

    return U_t

end



""" Symmetric version (Murg) of Murg exp(+i Jdt*XX ) """
function build_expXX_murg(sites::Vector{<:Index}, Jdt::Number; build_expZZ::Bool=false)

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    N = length(sites)
    U_XX = MPO(N)

     link_indices = hasqns(sites) ?
        [Index([QN("SzParity", 1, 2) => 1, QN("SzParity", 0, 2) => 1], "Link,l=$(n-1)", dir=ITensors.In) for n = 1:N+1] : 
        [Index(2, "Link,l=$(n-1)") for n = 1:N+1]

    for n = 1:N
        # siteindex s

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(link_indices[n])
        # right link index rl
        rl = link_indices[n+1]

        I = op(sites, "Id", n) 
        X = build_expZZ ? op(sites, "Z", n) : op(sites, "X", n)

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



""" Fourth-order Trotter MPO for Murg exp(-i*H*dt) Ising 
Convention H = -( JXX + gzZ + λxX )
"""

function build_expH_ising_murg_4o(
    sites::Vector{<:Index},
    JXX::Real,
    gz::Real,
    λx::Real,
    dt::Number)

    tfac = 2^(1/3)
    dt1 = dt/(2-tfac)
    dt2 = -dt*tfac/(2-tfac)

    U1 = build_expH_ising_murg_new(sites, JXX, gz, λx, dt1)
    U2 = build_expH_ising_murg_new(sites, JXX, gz, λx, dt2)

    U4 = applyn(U2, U1)
    U4 = applyn(U1, U4)

    return U4
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



""" Floquet Ising exp(-iJXX - iλX)exp(-igZ) """
function build_expH_ising_floquet(sites::Vector{<:Index}, JXX::Real, gz::Real, λx::Real; dt=1.0)

    Uxx = build_expXX_murg(sites, -JXX*dt)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*λx*dt) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*gz*dt) for s in sites])

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λx) ? Uxx : applyn(Ux, Uxx) 
    U_t = iszero(gz) ? U_t : applyn(Uz, U_t) 

    return U_t

end

""" Floquet Ising exp(-iJZZ - iλZ)exp(-igX) """
function build_expHZZ_ising_floquet(sites::Vector{<:Index}, JXX::Real, gperp::Real, λpar::Real; dt=1.0)

    Uzz = build_expXX_murg(sites, -JXX*dt; build_expZZ=true)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*gperp*dt) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*λpar*dt) for s in sites])

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λpar) ? Uxx : applyn(Uz, Uzz) 
    U_t = iszero(gperp) ? U_t : applyn(Ux, U_t) 

    return U_t

end


# Boilerplate 

function build_expH_ising_murg_new(s::Vector{<:Index}, p::IsingParams, dt::Number)
    build_expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end

function build_expH_ising_murg_new(p::IsingParams, dt::Number)
    s = siteinds("S=1/2", 3)
    build_expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end


function build_expH_ising_murg(s::Vector{<:Index}, p::IsingParams, dt::Number)
    build_expH_ising_murg(s, p.Jtwo, p.gperp, p.hpar, dt)
end

function build_expH_ising_murg(mp::IsingParams, dt::Number)
    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    build_expH_ising_murg(space_sites, mp.Jtwo, mp.gperp, mp.hpar, dt)

end
function build_expH_ising_murg_4o(p::IsingParams, dt::Number)
    s = siteinds("S=1/2", 3)
    build_expH_ising_murg_4o(s, p, dt)
end

function build_expH_ising_murg_4o(s::Vector{<:Index}, p::IsingParams, dt::Number)
    build_expH_ising_murg_4o(s, p.Jtwo, p.gperp, p.hpar, dt)
end

build_expHZZ_ising_floquet(s::Vector{<:Index}, p::IsingParams; dt::Number) =   build_expHZZ_ising_floquet(s, p.Jtwo, p.gperp, p.hpar; dt)

function build_expHZZ_ising_floquet(p::IsingParams; dt::Number)
    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    build_expHZZ_ising_floquet(space_sites, p.Jtwo, p.gperp, p.hpar; dt)
end
