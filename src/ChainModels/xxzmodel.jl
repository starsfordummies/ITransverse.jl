""" XXZ Spin 1/2 model - TODO: Update to use XXZParams """

""" TODO Haven't checked this in a while, tread with care"""
function build_H_XXZ_manual(sites, JXX::Real, hz::Real )
    """ Builds manually (no autompo) H ising Hamiltonian, convention 
    H = -( JXX + hZ ) 
    specify J and h as input params 
    """

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

    L = onehot(linkindices[1] => 1)
    R = onehot(dag(linkindices[N+1]) => 3)


    H[1] *= L
    H[N] *= R

    return H
end




""" Builds with autompo H XX Hamiltonian, convention 
H = -( J(XX+YY) + 2*hZ ) 
specify JXX and hZ as input params 
"""
function build_H_XX(sites, JXX::Real, hz::Real)

    build_H_XXZ(sites, JXX, 0, hz)
end

""" Builds with autompo H XXZ Hamiltonian, convention 
H = -( J(XX+YY+ Δ*ZZ) + 2*hZ ) 
specify JXX, ΔZZ and hZ as input params 
"""
function build_H_XXZ(sites, JXX::Real, ΔZZ::Real, hz::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -JXX,     "X", j, "X", j + 1
        os += -JXX,     "Y", j, "Y", j + 1
        if abs(ΔZZ) > 1e-10 
            os += -JXX*ΔZZ, "Z", j, "Z", j + 1
        end
    end

    if abs(hz) > 1e-10
        for j in 1:N
            os += -2*hz, "Z", j
        end
    end

    # Convert these terms to MPO
    return MPO(os, sites)
end



""" Builds with autompo H XXZ Hamiltonian using S+ and S- operators, convention 
H = -( J(XX+YY+ Δ*ZZ) + 2*hZ ) 
specify JXX, ΔZZ and hZ as input params 
"""
function build_H_XXZ_SpSm(sites, JXX::Real, ΔZZ::Real, hz::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -JXX/2,     "S+", j, "S-", j + 1
        os += -JXX/2,     "S-", j, "S+", j + 1
        if abs(ΔZZ) > 1e-10 
            os += -JXX*ΔZZ, "Z", j, "Z", j + 1
        end
    end

    if abs(hz) > 1e-10
        for j in 1:N
            os += -2*hz, "Z", j
        end
    end

    # Convert these terms to MPO
    return MPO(os, sites)
end





# !This is not implemented yet 
# """ Symmetric version of Murg exp(-i*H_XXZ*t) """
# function build_expH_XXZ_murg(
#     sites,
#     JXX::Real,
#     ΔZZ::Real,
#     hz::Real,
#     dt::Number)


#     @assert ΔZZ < 0.001  # for now 

#     # For real dt this does REAL time evolution 
#     # I should have already taken into account both the - sign in exp(-iHt) 
#     # and the overall minus in Ising H= -(JXX+Z)

#     dt = JXX*dt

#     cosg = cos(hz*dt*0.5)
#     sing = sin(hz*dt*0.5)


#     N = length(sites)
#     U_t = MPO(N)

#     link_dimension = 2

#     linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]


#     for n = 1:N
#         # siteindex s
#         s = sites[n]

#         # left link index ll with daggered QN conserving direction (if applicable)
#         ll = dag(linkindices[n])
#         # right link index rl
#         rl = linkindices[n+1]

#         combz = (1 - 2*sing^2)*op(sites, "Id", n) + im*2*sing*cosg*op(sites, "Z", n)
#         X = op(sites, "X", n)

#         if n == 1
#             #U_t[n] = ITensor(ComplexF64, dag(s), s', dag(rl))
#             U_t[n] = onehot(rl => 1) * sqrt(cos(dt))*combz
#             U_t[n] += onehot(rl => 2) * sqrt(im*sin(dt))*X
#         elseif n == N
#             #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
#             U_t[n] = onehot(ll => 1) * sqrt(cos(dt))*combz
#             U_t[n] += onehot(ll => 2) * sqrt(im*sin(dt))*X

#         else
#             #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

#             U_t[n] = onehot(ll => 1, rl =>1) * cos(dt)*combz
#             U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
#             U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
#             U_t[n] += onehot(ll => 2, rl =>2) * im*sin(dt)*combz
#         end


#     end

#     return U_t


# end



""" Symmetric version of Murg exp(-i*H_XX*t) built as product of three Ising MPO W's """
function build_expH_XX_murg_from_ising(
    space_sites,
    JXX::Real,
    dt::Number)


    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)


    N = length(space_sites)
    U_t = MPO(N)


    linkindices = [Index(8, "Link,l=$(n-1)") for n = 1:N+1]

    dt = JXX*dt

    eH_YY =  build_expH_ising_murg_YY(space_sites, 1, dt/2)
    eH_XX = build_expH_ising_murg(space_sites, 1, 0, dt)


    temp1 = eH_YY[1] * eH_XX[1]' * eH_YY[1]''
    comb_R = combiner(inds(temp1,"Link"))
    temp_left = temp1 * comb_R * delta( inds(comb_R)[1], linkindices[2] )
    temp_left = replaceprime(temp_left, 3=>1)

    U_t[1] = temp_left 


    for nn = 2:N-1
    temp1 = eH_YY[nn] * eH_XX[nn]' * eH_YY[nn]''
    comb_L = combiner(inds(temp1,"l="*string(nn-1)))
    comb_R = combiner(inds(temp1,"l="*string(nn)))
    temp_center = temp1 * comb_L * comb_R * delta( inds(comb_L)[1], linkindices[nn] ) *  delta( inds(comb_R)[1], linkindices[nn+1] )
    temp_center = replaceprime(temp_center, 3=>1)

    U_t[nn] = temp_center
    end

    temp1 = eH_YY[end] * eH_XX[end]' * eH_YY[end]''
    #println(temp1)
    comb_L = combiner(inds(temp1,"Link"))
    temp_right = temp1 * comb_L  * delta( inds(comb_L)[1], linkindices[end-1] ) 
    temp_right = replaceprime(temp_right, 3=>1)

    U_t[end] = temp_right


    return U_t


end


""" Symmetric version of Murg exp(-i*H_XX*t) built as product of three Ising MPO W's
 using H = -J(XX + ZZ) convention """
function build_expH_XXZZ_murg_from_ising(
    space_sites,
    JXX::Real,
    dt::Number)


    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)


    N = length(space_sites)
    U_t = MPO(N)


    linkindices = [Index(8, "Link,l=$(n-1)") for n = 1:N+1]

    dt = JXX*dt

    eH_ZZ =  build_expH_ising_murg_ZZX(space_sites, 1, 0., dt/2)
    eH_XX = build_expH_ising_murg(space_sites, 1, 0, dt)


    temp1 = eH_ZZ[1] * eH_XX[1]' * eH_ZZ[1]''
    comb_R = combiner(inds(temp1,"Link"))
    temp_left = temp1 * comb_R * delta( inds(comb_R)[1], linkindices[2] )
    temp_left = replaceprime(temp_left, 3=>1)

    U_t[1] = temp_left 


    for nn = 2:N-1
    temp1 = eH_ZZ[nn] * eH_XX[nn]' * eH_ZZ[nn]''
    comb_L = combiner(inds(temp1,"l="*string(nn-1)))
    comb_R = combiner(inds(temp1,"l="*string(nn)))
    temp_center = temp1 * comb_L * comb_R * delta( inds(comb_L)[1], linkindices[nn] ) *  delta( inds(comb_R)[1], linkindices[nn+1] )
    temp_center = replaceprime(temp_center, 3=>1)

    U_t[nn] = temp_center
    end

    temp1 = eH_ZZ[end] * eH_XX[end]' * eH_ZZ[end]''
    #println(temp1)
    comb_L = combiner(inds(temp1,"Link"))
    temp_right = temp1 * comb_L  * delta( inds(comb_L)[1], linkindices[end-1] ) 
    temp_right = replaceprime(temp_right, 3=>1)

    U_t[end] = temp_right


    return U_t


end


""" TODO CHECK Symmetric version of Murg exp(-i*H_XXZ*t) built as product of three Ising MPO W's """
function build_expH_XXZ_murg_from_ising(
    space_sites,
    JXX::Real,
    hz::Real,
    dt::Number)

    # For real dt this does REAL time evolution 

    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)


    N = length(space_sites)
    U_t = MPO(N)


    linkindices = [Index(8, "Link,l=$(n-1)") for n = 1:N+1]

    dt = JXX*dt

    eH_YY =  build_expH_ising_murg_YY(space_sites, 1, dt/2)
    eH_XX = build_expH_ising_murg(space_sites, 1, 0, 0, dt)


    temp1 = eH_YY[1] * eH_XX[1]' * eH_YY[1]''
    comb_R = combiner(inds(temp1,"Link"))
    temp_left = temp1 * comb_R * delta( inds(comb_R)[1], linkindices[2] )
    temp_left = replaceprime(temp_left, 3=>1)

    U_t[1] = temp_left 


    for nn = 2:N-1
    temp1 = eH_YY[nn] * eH_XX[nn]' * eH_YY[nn]''
    comb_L = combiner(inds(temp1,"l="*string(nn-1)))
    comb_R = combiner(inds(temp1,"l="*string(nn)))
    temp_center = temp1 * comb_L * comb_R * delta( inds(comb_L)[1], linkindices[nn] ) *  delta( inds(comb_R)[1], linkindices[nn+1] )
    temp_center = replaceprime(temp_center, 3=>1)

    U_t[nn] = temp_center
    end

    temp1 = eH_YY[end] * eH_XX[end]' * eH_YY[end]''
    #println(temp1)
    comb_L = combiner(inds(temp1,"Link"))
    temp_right = temp1 * comb_L  * delta( inds(comb_L)[1], linkindices[end-1] ) 
    temp_right = replaceprime(temp_right, 3=>1)

    U_t[end] = temp_right


    hz = 2*hz 
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



""" exp(-i*H_XX*t) using symmetric SVD (??) and Splus-Sminus convention
FIXME this doesnt work 
"""
function build_expH_XX_SpSm_svd(
    in_space_sites,
    JXX::Real,
    dt::Number)


    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)


    N = length(in_space_sites)
    U_t = MPO(N)

    ϵ = JXX * 1.0im * dt 

    uT_open = ITensor()

    for n = 1:N-1 # TODO CHECK THIS
    
        Spi = op(in_space_sites, "S+", n)
        Smi = op(in_space_sites, "S-", n)
    
        Spj = op(in_space_sites, "S+", n+1)
        Smj = op(in_space_sites, "S-", n+1)
    
    
        e1 = exp(2*ϵ*(Spi * Smj + Smi * Spj))
    
        c1 = combiner(inds(Spi))
        c2 = combiner(inds(Spj))
    
        e1c = e1 * c1 * c2

        #@show e1c
        #@show matrix(e1c)
        
        u, s, uT, _, _ = symm_svd(e1c, combinedind(c1), cutoff=1e-15)

        u_sqs = u * sqrt.(s)
        uT_sqs = sqrt.(s) * uT

        u_open = u_sqs * dag(c1) * delta(inds(s))
        replacetags!(u_open, "u" => "Link,l=$n")


        if n == 1
            U_t[n] = u_open
            
        else
            uu = uT_open * prime(u_open, "Site")
            uu = replaceprime( uu, 2 => 1)
            U_t[n] = uu
        end

        uT_open = uT_sqs * dag(c2)
        replacetags!(uT_open, "u" => "Link,l=$n")


    end # for n = 1:N-1

    U_t[N] = uT_open


    @show U_t

    return U_t


end



""" exp(-i*H_XX*t) using symmetric SVD - TODO Check """
function build_expH_XX_svd(
    in_space_sites,
    JXX::Real,
    dt::Number)


    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)


    N = length(in_space_sites)
    U_t = MPO(N)


    ϵ = JXX * 1.0im * dt 

    uT_open = ITensor()

    for n = 1:N-1 # TODO CHECK THIS
    
        Xi = op(in_space_sites, "X", n)
        Yi = op(in_space_sites, "Y", n)
    
        Xj = op(in_space_sites, "X", n+1)
        Yj = op(in_space_sites, "Y", n+1)
    
    
        e1 = exp(ϵ*(Xi * Xj + Yi * Yj))
    
        c1 = combiner(inds(Xi))
        c2 = combiner(inds(Xj))
    
        e1c = e1 * c1 * c2

        #@show e1c
        #@show matrix(e1c)

        u, s, uT, _, _ = symm_svd(e1c, combinedind(c1), cutoff=1e-15)


        u_sqs = u * sqrt.(s)
        uT_sqs = sqrt.(s) * uT

        u_open = u_sqs * dag(c1) * delta(inds(s))
        replacetags!(u_open, "u" => "Link,l=$n")


        if n == 1
            U_t[n] = u_open
            
        else
            uu = uT_open * prime(u_open, "Site")
            uu = replaceprime( uu, 2 => 1)
            U_t[n] = uu
        end

        uT_open = uT_sqs * dag(c2)
        replacetags!(uT_open, "u" => "Link,l=$n")


    end # for n = 1:N-1

    
    U_t[N] = uT_open
    

    return U_t


end
