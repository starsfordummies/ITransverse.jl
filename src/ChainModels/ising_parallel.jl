############ Transverse field Ising ##############
###### Our convention is usually H = -JXX - gZ - hX 
######## Hamiltonian ########


""" Builds Ising Hamiltonian MPO  H = -Jtwo*XX - gperp*Z - hpar*X """ 
function H_ising(sites::Vector{<:Index}, Jtwo::Real, gperp::Real, hpar::Real)

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
- For exp(-iHdt) dt must be either included already in the parameters
"""
function expH_ising_murg(sites::Vector{<:Index}, JXX::Number, gz::Number, λx::Number)
    """ Symmetric version of Murg exp(-iHising t) """


    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    Uxx = expXX_murg(sites, JXX)

    Ux = MPO([op(s, "Rx", θ=-2*λx) for s in sites])
    Uz2 = MPO([op(s, "Rz", θ=-gz) for s in sites])


    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λx) ? Uz2 : applyn(Ux, Uz2) 
    U_t = applyn(Uxx, U_t) 
    U_t = applyn(Uz2, U_t) 

    return U_t

end


""" Symmetric version (Murg) of Murg exp(+i Jdt*XX ) """
function expXX_murg(sites::Vector{<:Index}, Jdt::Number; make_expZZ::Bool=false)

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
        X = make_expZZ ? op(sites, "Z", n) : op(sites, "X", n)

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





function expH_ising_symm_svd(s::Vector{<:Index}, Jtwodt::Number, hperpdt::Number, λpardt::Number)
    w = expH_ising_symm_svd_3site(Jtwodt, hperpdt, λpardt)
    wmpo = if length(s) == 3
        replace_siteinds(w, s)
    else
        extend_mpo(s, w)
    end
    return wmpo
end


""" Builds core MPO tensors for 3 sites """ 
function expH_ising_symm_svd_3site(Jtwodt::Number, hperpdt::Number, λpardt::Number)

    s = siteinds("S=1/2", 3)

    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    e12 = exp(im*X1*X2*Jtwodt)
    e23 = exp(im*X2*X3*Jtwodt)

    fac_z = hperpdt*0.5
    eZ1 = exp(im*fac_z*op(s,"Z",1))
    eZ2 = exp(im*fac_z*op(s,"Z",2))
    eZ3 = exp(im*fac_z*op(s,"Z",3))

    eX1 = exp(im*λpardt*op(s,"X",1))
    eX2 = exp(im*λpardt*op(s,"X",2))
    eX3 = exp(im*λpardt*op(s,"X",3))

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





""" Fourth-order Trotter MPO for Murg exp(-i*H*dt) Ising 
Convention H = -( JXX + gzZ + λxX )
"""

function expH_ising_murg_4o(
    sites::Vector{<:Index},
    Jtwodt::Number,
    gperpdt::Number,
    λpardt::Number)

    tfac = 2^(1/3)
    dt1 = 1/(2-tfac)
    dt2 = -tfac/(2-tfac)

    U1 = expH_ising_murg(sites, Jtwodt*dt1, gperpdt*dt1, λpardt*dt1)
    U2 = expH_ising_murg(sites, Jtwodt*dt2, gperpdt*dt2, λpardt*dt2)

    U4 = applyn(U2, U1)
    U4 = applyn(U1, U4)

    return U4
end




""" Floquet Ising exp(-iJXX - iλX)exp(-igZ) """
function expH_ising_floquet(sites::Vector{<:Index}, JXX::Number, gz::Number, λx::Number)

    Uxx = expXX_murg(sites, -JXX)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*λx) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*gz) for s in sites])

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λx) ? Uxx : applyn(Ux, Uxx) 
    U_t = iszero(gz) ? U_t : applyn(Uz, U_t) 

    return U_t

end

""" Floquet Ising exp(-iJZZ - iλZ)exp(-igX) """
function expHZZ_ising_floquet(sites::Vector{<:Index}, JXX::Number, gperp::Number, λpar::Number)

    Uzz = expXX_murg(sites, -JXX; make_expZZ=true)

    # Recall Ra(theta) = exp(-i sigma_a(theta/2))
    Ux = MPO([op(s, "Rx", θ=2*gperp) for s in sites])
    Uz = MPO([op(s, "Rz", θ=2*λpar) for s in sites])

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(λpar) ? Uzz : applyn(Uz, Uzz) 
    U_t = iszero(gperp) ? U_t : applyn(Ux, U_t) 

    return U_t

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




#= 
# Boilerplate 


function H_ising(sites::Vector{<:Index}, mp::IsingParams)
    H_ising(sites, mp.Jtwo, mp.gperp, mp.hpar)
end


function expH_ising_murg_new(s::Vector{<:Index}, p::IsingParams, dt::Number)
    expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end

function expH_ising_murg_new(p::IsingParams, dt::Number)
    s = siteinds("S=1/2", 3)
    expH_ising_murg_new(s, p.Jtwo, p.gperp, p.hpar, dt)
end


function expH_ising_murg(s::Vector{<:Index}, p::IsingParams, dt::Number)
    expH_ising_murg(s, p.Jtwo, p.gperp, p.hpar, dt)
end

function expH_ising_murg(mp::IsingParams, dt::Number)
    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    expH_ising_murg(space_sites, mp.Jtwo, mp.gperp, mp.hpar, dt)

end
function expH_ising_murg_4o(p::IsingParams, dt::Number)
    s = siteinds("S=1/2", 3)
    expH_ising_murg_4o(s, p, dt)
end

function expH_ising_murg_4o(s::Vector{<:Index}, p::IsingParams, dt::Number)
    expH_ising_murg_4o(s, p.Jtwo, p.gperp, p.hpar, dt)
end

expHZZ_ising_floquet(s::Vector{<:Index}, p::IsingParams, dt::Number) =   expHZZ_ising_floquet(s, p.Jtwo, p.gperp, p.hpar, dt)

function expHZZ_ising_floquet(p::IsingParams, dt::Number)
    space_sites = siteinds("S=1/2", 3; conserve_qns = false)
    expHZZ_ising_floquet(space_sites, p.Jtwo, p.gperp, p.hpar, dt)
end

=# 