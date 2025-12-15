using ITransverse.ITenUtils: symm_svd


""" Builds with autompo H XXZ Hamiltonian, convention 
H = -( J(XX+YY+ Δ*ZZ) + 2*hZ ) 
specify JXX, ΔZZ and hZ as input params 
"""
function build_H_XXZ(sites, JXY::Real, ΔZZ::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -JXY,     "Sx", j, "Sx", j + 1
        os += -JXY,     "Sy", j, "Sy", j + 1
        if abs(ΔZZ) > 1e-10 
            os += -JXY*ΔZZ, "Sz", j, "Sz", j + 1
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
            os += -JXX*ΔZZ, "Sz", j, "Sz", j + 1
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
function build_H_XXZ_SpSm(sites, mp::XXZParams)
    build_H_XXZ_SpSm(sites, mp.J_XY, mp.J_ZZ, mp.hz)
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

# Boilerplate

function build_H_XXZ(sites, mp::XXZParams)
    build_H_XXZ(sites, mp.J_XY, mp.J_ZZ)
end

function build_expH_XXZ_2o_spin1(p,dt) 
    s = siteinds("S=1", 3)
    build_expH_XXZ_2o(s, p,dt) 
end

function build_expH_XXZ_2o_spinhalf(p,dt) 
    s = siteinds("S=1/2", 3)
    build_expH_XXZ_2o(s, p,dt) 
end

build_expH_XXZ_2o(sites, p,dt) = timeEvo_MPO_2ndOrder(sites, fill("Id", 3), zeros(3), ["S+", "S-", "Sz"], [0.5*p.J_XY, 0.5*p.J_XY, p.J_ZZ], ["S-", "S+", "Sz"], ones(3), "Sz", p.hz, dt)
