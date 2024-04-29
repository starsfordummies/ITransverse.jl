#using ITensors
#using LongRangeITensors


""" Builds manually (no autompo) H ising Hamiltonian, convention 
H = -( JXX + hZ ) 
specify J and h as input params 
"""

function build_H_ising_manual(sites, JXX::Real, hz::Real )

    # link_dimension
    link_dimension = 3
    
    N = length(sites)

    # hasqns(sites) ? error("The transverse field Ising model does not conserve total Spin Z") : true

    # generate "regular" link indeces (i.e. without conserved QNs)
    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    H = MPO(sites)



    """ use UPPER triangular convention:  
        | 1 C D  |                             0          
    H = |   A B  |   ,   L = [1 0 0] ,    R =  0
        |     1  |                             1
    """

    for n = 1:N
        # siteindex s
        s = sites[n]
        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]
    
        # init empty ITensor 
        H[n] = ITensor(Float64, ll, dag(s), s', rl)

        Id = op(sites, "Id", n)
        CC = op(sites, "X",  n)
        BB = op(sites, "X",  n) * (- JXX)
        DD = op(sites, "Z",  n) * (- hz)
        #AA = 0.
        if n == 1
            println("UPTRI inputs: $JXX and $hz")
            println("BB = $(BB)")
            println("CC = $(CC)")
            end
        # add both Identities as netral elements in the MPS at corresponding location (setelement function)
        H[n] += onehot(ll => 1, rl => 1) * Id
        H[n] += onehot(ll => 1, rl => 2) * CC # σˣ
        H[n] += onehot(ll => 1, rl => 3) * DD # hz σᶻ
        H[n] += onehot(ll => 2, rl => 3) * BB # Jxx σˣ
        H[n] += onehot(ll => 3, rl => 3) * Id


        #H[n] += onehot(ll =>2 , rl => 2) * op(sites, "Id", n) * λ  # λ Id,  on the diagonal
    end

    # project out the left and right boundary MPO with unit row/column vector
    #L = ITensor(linkindices[1])
    #L[startState] = 1.0
    # R = ITensor(dag(linkindices[N+1]))
    # R[endState] = 1.0

    L = onehot(linkindices[1] => 1)
    R = onehot(dag(linkindices[N+1]) => 3)


    H[1] *= L
    H[N] *= R

    return H
end


""" Builds manually (no autompo) H ising Hamiltonian, convention 
H = -( JXX + hZ ) 
specify J and h as input params 
Just an alternative version using the lower-triangular form - should be identical to _manual
"""
function build_H_ising_manual_lowtri(sites, JXX::Real, hz::Real )

    # link_dimension
    link_dimension = 3
    
    N = length(sites)

    # hasqns(sites) ? error("The transverse field Ising model does not conserve total Spin Z") : true

    # generate "regular" link indeces (i.e. without conserved QNs)
    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    H = MPO(sites)



    """ use LOWER triangular convention:  
        | 1     |                             1          
    H = | C A   |   ,   L = [0 0 1] ,    R =  0
        | D B 1 |                             0
    """

    for n = 1:N
        # siteindex s
        s = sites[n]
        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]
    
        # init empty ITensor 
        H[n] = ITensor(Float64, ll, dag(s), s', rl)

        Id = op(sites, "Id", n)
        CC = op(sites, "X",  n)
        BB = op(sites, "X",  n) * (- JXX)
        DD = op(sites, "Z",  n) * (- hz)
        #AA = 0.
        if n == 1
        println("LOWTRI inputs: $JXX and $hz")
        println("BB = $(BB)")
        println("CC = $(CC)")
        end

        # add both Identities as netral elements in the MPS at corresponding location (setelement function)
        H[n] += onehot(ll => 1, rl => 1) * Id
        H[n] += onehot(ll => 2, rl => 1) * CC # σˣ
        H[n] += onehot(ll => 3, rl => 1) * DD # hz σᶻ
        H[n] += onehot(ll => 3, rl => 2) * BB # Jxx σˣ
        H[n] += onehot(ll => 3, rl => 3) * Id


        #H[n] += onehot(ll =>2 , rl => 2) * op(sites, "Id", n) * λ  # λ Id,  on the diagonal
    end


    L = onehot(linkindices[1] => 3)
    R = onehot(dag(linkindices[N+1] => 1))


    H[1] *= L
    H[N] *= R

    return H
end


""" Builds with autompo H ising Hamiltonian, convention 
H = -( JXX + hZ ) 
specify J and h as input params 
"""
function build_H_ising(sites, Jxx::Real, hz::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -Jxx, "X", j, "X", j + 1
    end

    for j in 1:N
        os += -hz, "Z", j
    end

    # Convert these terms to an MPO tensor network
    return MPO(os, sites)
end

""" Builds with autompo H ising Hamiltonian, convention 
H = -( JZZ + hX ) 
specify J and h as input params 
"""
function build_H_ising_ZZ_X(sites, Jzz::Real, hx::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -Jzz, "Z", j, "Z", j + 1
    end

    for j in 1:N
        os += -hx, "X", j
    end

    # Convert these terms to an MPO tensor network
    return MPO(os, sites)
end



""" Builds with autompo H ising Hamiltonian, convention 
H = -( JYY + hZ ) 
specify J and h as input params 
"""
function build_H_ising_YY(sites, Jyy::Real, hz::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -Jyy, "Y", j, "Y", j + 1
    end

    for j in 1:N
        os += -hz, "Z", j
    end

    # Convert these terms to an MPO tensor network
    return MPO(os, sites)
end



##############################################################
##############################################################
########    exponentials of H Ising for time evolution   #####
##############################################################
##############################################################


""" Second order approximation for exp(-i*H*dt) Ising using Maartens et al prescription 
Convention H = -( JXX + hZ )
"""
function build_expH_ising_2o(
    sites,
    JXX::Real, hz::Real,
    dt::Number;
    )
    """ Second-order approximation for exp(-iHising t) from VanDamme et al """

    # measure τ in imaginary units for actualy time evolution
    τ = -1.0im * dt
    N = length(sites)
    U_t = MPO(N)

    link_dimension = 3
    # startState = 3
    # endState = 1

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    # loop over real space finite-sized MPO and fill (here) homogeniously
    for n = 1:N
        # siteindex s
        s = sites[n]
        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]
        Id = op(sites, "Id", n)
    
        # A is possible exponential decay so test for "0"
        A =  0.0 * op(sites, "Id", n) 
        B = -JXX * op(sites, "X", n)
        C =  op(sites, "X", n)
        D = -hz * op(sites, "Z", n)


        # Init ITensor inside MPO
        #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
        # first row
        U_t[n] = onehot(ll => 1, rl => 1) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) + (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1))
        U_t[n] += onehot(ll => 1, rl => 2) * (C + (τ / 2) * braket(C, D) + (τ^2 / 6) * braket2(C, D))
        U_t[n] += onehot(ll => 1, rl => 3) * (replaceprime(C' * C, 2, 1) + (τ / 3) * braket2(D, C))
        # second row
        U_t[n] += onehot(ll => 2, rl => 1) * (τ * B + (τ^2 / 2) * braket(B, D) + (τ^3 / 6) * braket2(B, D))
        U_t[n] += onehot(ll => 2, rl => 2) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) + (τ^2 / 6) * (braket(C, B, D) + braket2(A, D)))
        U_t[n] += onehot(ll => 2, rl => 3) * (braket(A, C) + (τ / 3) * (braket(A, C, D) + braket2(B, C)))
        # third row
        U_t[n] += onehot(ll => 3, rl => 1) * ((τ^2 / 2) * replaceprime(B' * B, 2, 1) + (τ^3 / 6) * braket2(D, B))
        U_t[n] += onehot(ll => 3, rl => 2) * ((τ / 2) * braket(A, B) + (τ^2 / 6) * (braket(A, B, D) + braket2(C, B)))
        U_t[n] += onehot(ll => 3, rl => 3) * (replaceprime(A' * A, 2, 1) + (τ / 3) * (braket(A, B, C) + braket2(D, A)))
    end

    # implementing OBC: project out upper row and fist column for right and left boundaries, respectively
    L = onehot(linkindices[1] => 1)
    R = onehot(dag(linkindices[N+1] => 1))

    U_t[1] *= L
    U_t[N] *= R

    return U_t
end


""" First-order approximation for exp(-i*H*dt) Ising using Maartens et al prescription 
Convention H = -( JXX + hZ )
"""
function build_expH_ising_1o(
    sites,
    JXX::Real, hz::Real,
    dt::Number;
    )
    """ Builds 1st order approx for exp(Hising) from Van Damme et al
    The Hamiltonian is H = -(XX + Z)  watch the sign! 
    """

    # measure τ in imaginary units for actualy time evolution
    τ = -1.0im * dt
    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2
 
    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    for n = 1:N
        # siteindex s
        #s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]
        Id = op(sites, "Id", n)

        # A is possible exponential decay so test for "0"
        A =  0.0 * op(sites, "Id", n) 
        B = -JXX * op(sites, "X", n)
        C =  op(sites, "X", n)
        D = -hz * op(sites, "Z", n)

        # Init ITensor inside MPO
        #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
        # first row
        U_t[n] = onehot(ll => 1, rl => 1) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) )
        U_t[n] += onehot(ll => 1, rl => 2) * (C + (τ / 2) * braket(C, D))
        # second row
        U_t[n] += onehot(ll => 2, rl => 1) * (τ * B + (τ^2 / 2) * braket(B, D))
        U_t[n] += onehot(ll => 2, rl => 2) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) )
    end

    L = onehot(linkindices[1] => 1)
    R = onehot(dag(linkindices[N+1] => 1))

    U_t[1] *= L
    U_t[N] *= R

    return U_t
end


""" Prescription a la Murg for exp(-i*H*dt) Ising  - gives symmetric MPO matrices
Convention H = -( JXX + hZ )
"""
function build_expH_ising_murg(sites::Vector{<:Index}, p::pparams)
    build_expH_ising_murg(sites, p.JXX, p.hz, p.dt)
end

""" Prescription a la Murg for exp(-i*H*dt) Ising  
Convention H = -( JXX + hZ )
"""
function build_expH_ising_murg(
    sites::Vector{<:Index},
    JXX::Real,
    hz::Real,
    dt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    dt = JXX*dt

    cosg = cos(hz*dt*0.5)
    sing = sin(hz*dt*0.5)


    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]


    for n = 1:N
        # siteindex s
        s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        combz = (1 - 2*sing^2)*op(sites, "Id", n) + im*2*sing*cosg*op(sites, "Z", n)
        X = op(sites, "X", n)

        if n == 1
            #U_t[n] = ITensor(ComplexF64, dag(s), s', dag(rl))
            U_t[n] = onehot(rl => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(rl => 2) * sqrt(im*sin(dt))*X
        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] = onehot(ll => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(ll => 2) * sqrt(im*sin(dt))*X

        else
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_t[n] = onehot(ll => 1, rl =>1) * cos(dt)*combz
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>2) * im*sin(dt)*combz
        end


    end

    return U_t


end



""" Symmetric version of Murg exp(-iHising t) \\
with the convention H = -(ZZ + X) !!  """
function build_expH_ising_murg_ZZX(
    sites,
    JXX::Real,
    hz::Real,
    dt::Number)

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    dt = JXX*dt

    cosg = cos(hz*dt*0.5)
    sing = sin(hz*dt*0.5)


    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]


    for n = 1:N
        # siteindex s
        s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        combz = (1 - 2*sing^2)*op(sites, "Id", n) + im*2*sing*cosg*op(sites, "X", n)
        X = op(sites, "Z", n)

        if n == 1
            #U_t[n] = ITensor(ComplexF64, dag(s), s', dag(rl))
            U_t[n] = onehot(rl => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(rl => 2) * sqrt(im*sin(dt))*X
        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] = onehot(ll => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(ll => 2) * sqrt(im*sin(dt))*X

        else
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_t[n] = onehot(ll => 1, rl =>1) * cos(dt)*combz
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>2) * im*sin(dt)*combz
        end


    end

    return U_t


end



""" Symmetric version of Murg exp(-iHising t) \\
with the convention H = -(YY) !!  """
function build_expH_ising_murg_YY(
    space_sites,
    JXX::Real,
    dt::Number)

    build_expH_ising_murg_YY(space_sites, JXX, 0., dt)
end

function build_expH_ising_murg_YY(
    space_sites,
    JXX::Real,
    hz::Real,
    dt::Number)

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    dt = JXX*dt

    N = length(space_sites)
    U_t = MPO(N)

    link_dimension = 2

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]


    for n = 1:N
        # siteindex s
        s = space_sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        combz = op(space_sites, "Id", n) 
        Y = op(space_sites, "Y", n)

        if n == 1
            #U_t[n] = ITensor(ComplexF64, dag(s), s', dag(rl))
            U_t[n] = onehot(rl => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(rl => 2) * sqrt(im*sin(dt))*Y
        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] = onehot(ll => 1) * sqrt(cos(dt))*combz
            U_t[n] += onehot(ll => 2) * sqrt(im*sin(dt))*Y

        else
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_t[n] = onehot(ll => 1, rl =>1) * cos(dt)*combz
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(dt))*sqrt(cos(dt))*Y
            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(dt))*sqrt(cos(dt))*Y
            U_t[n] += onehot(ll => 2, rl =>2) * im*sin(dt)*combz
        end


    end

    if hz > 1e-10
        for n = 1:N
    
            Z = op(space_sites, "Z",  n)
            expT = exp(im*dt * Z * hz/2)
    
            U_t[n] = prime(U_t[n], "Site") * expT
            U_t[n] = noprime(U_t[n] * prime(expT, "Site"), 2)
        end
    end

    return U_t


end

#= 
"""
Builds exp(Hpotts) with 2nd order approximation from the Ghent group.
Bond dimension is 7 
"""
function build_expH_ising_2o_Jan(sites, 
    J::Real, f::Real,
    dt::Number)


    #(sites,
    # AStrings::Vector{String}, JAs::Vector{<:Number}, 
    # BStrings::Vector{String}, JBs::Vector{<:Number}, 
    # CStrings::Vector{String}, JCs::Vector{<:Number}, 
    # DString::String, JD::Number, 
    # t::Number)

    U_t = timeEvo_MPO_2ndOrder(sites, 
    ["Id"], [0.], 
    ["X"], [-J],
    ["X"], [1.],
    "Z", -f,
    dt)

    return U_t
end
=#


function build_H_id(sites)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:N
        os += 1/N, "Id", j
    end

    # Convert these terms to an MPO tensor network
    return MPO(os, sites)
end