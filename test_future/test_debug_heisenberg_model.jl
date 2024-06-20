using ITensors
using ITensorTDVP


prodop(A::ITensor, B::ITensor) = replaceprime(prime(A) * B, 2, 1)
braket(A::ITensor, B::ITensor) = replaceprime(prime(A) * B + prime(B) * A, 2, 1)
anticomm(A::ITensor, B::ITensor) = replaceprime(A' * B + B' * A, 2,1)

braket(A::ITensor, B::ITensor, C::ITensor) = replaceprime(
    (A'' * B' * C) + (A'' * C' * B) + (B'' * A' * C) + (B'' * C' * A) + (C'' * A' * B) + (C'' * B' * A),
    3, 1)
braket2(A::ITensor, B::ITensor) = replaceprime((A'' * B' * B) + (B'' * A' * B) + (B'' * B' * A), 3, 1)



function build_H_heis_autompo(sites)
 
    os = OpSum()
    N = length(sites)

    for j=1:N-1
        os += 1,"S+",j,"S-",j+1
        os += 1,"S-",j,"S+",j+1
    end

    for j=1:N
        os += 1,"Z", j
    end

    H = MPO(os,sites)

end


function build_H_heis_manual(
    sites
    )::ITensors.MPO where {T1<:Real} where {T2<:Real}

    # Here I try to build manually the MPO for the Ham S+S- + hc + Z , 
    # should be something like 

    # 1 
    # S+
    # S- 
    # Z  S- S+ 1 

    # or rather (upper tri)

    # 1 S+ S- Z
    #         S-
    #         S+
    #         1

    # link_dimension
    d0 = dim(op(sites, "Id", 1), 1)
    link_dimension = 4
    
    startState = 1 # 4
    endState = 4

    N = length(sites)


    hasqns(sites) ? error("The transverse field Ising model does not conserve total Spin Z") : true

    # generate "regular" link indeces (i.e. without conserved QNs)
    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    H = MPO(sites)

    for n = 1:N
        # siteindex s
        s = sites[n]
        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]


        Id = op(sites, "Id", n)
        B1 = op(sites, "S-", n)
        B2 = op(sites, "S+", n)
        C1 = op(sites, "S+", n)
        C2 = op(sites, "S-", n)
        D =  op(sites, "Z",  n)

        # init empty ITensor with
        H[n] = ITensor(ComplexF64, ll, dag(s), s', rl)

        # add both Identities as netral elements in the MPS at corresponding location (setelement function)
        H[n] += setelt(ll[1]) * setelt(rl[1]) * Id
        H[n] += setelt(ll[1]) * setelt(rl[2]) * C1
        H[n] += setelt(ll[1]) * setelt(rl[3]) * C2
        H[n] += setelt(ll[1]) * setelt(rl[4]) * D

        H[n] += setelt(ll[2]) * setelt(rl[4]) * B1
        H[n] += setelt(ll[3]) * setelt(rl[4]) * B2
        H[n] += setelt(ll[4]) * setelt(rl[4]) * Id

    end

    # project out the left and right boundary MPO with unit row/column vector
    L = ITensor(linkindices[1])
    L[startState] = 1.0

    R = ITensor(dag(linkindices[N+1]))
    R[endState] = 1.0

    H[1] *= L
    H[N] *= R

    return H
end


function build_expH_heis_firstorder(sites, dt)

    N = length(sites)

    link_dimension = 3

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_t = MPO(N)


    τ = -1.0im * dt


    for n = 1:N
        # siteindex s
        s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        

        Id = op(sites, "Id", n)

        #A =  0.0 * op(sites, "Id", n)
        B1 = op(sites, "S-", n)
        B2 = op(sites, "S+", n)
        C1 = op(sites, "S+", n)
        C2 = op(sites, "S-", n)
        D =  op(sites, "Z",  n)

        B = [B1, B2]
        C = [C1, C2]

        # Init ITensor inside MPO
        U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))


        #[1,1]
        U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 4) * anticomm(D,D) )

        # rest of first row 
        for (iiC, Ci) in enumerate(C)
            U_t[n] += setelt(ll[1]) * setelt(rl[1+iiC]) * (Ci + (τ / 2) * anticomm(Ci, D))
            println("filling $(1), $(iiC+1)")

        end

        # rest of first column 
        for (iiB, Bi) in enumerate(B)
            U_t[n] += setelt(ll[1+iiB]) * setelt(rl[1]) * (τ * Bi + (τ^2 / 2) * anticomm(Bi, D))
            println("filling $(1+iiB), $(1)")

        end

        # bottom block
        for (iiC, Ci) in enumerate(C)
            for (iiB, Bi) in enumerate(B)
                U_t[n] += setelt(ll[1+iiB]) * setelt(rl[1+iiC]) * ( (τ / 2) * anticomm(Bi,Ci) )
                println("filling $(iiB+1), $(iiC+1)")
            end
        end


        # This should work 
        # # first row
        # U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 4) * anticomm(D,D) )
        # U_t[n] += setelt(ll[1]) * setelt(rl[2]) * (C1 + (τ / 2) * anticomm(C1, D))
        # U_t[n] += setelt(ll[1]) * setelt(rl[3]) * (C2 + (τ / 2) * anticomm(C2, D))

        # # second row
        # U_t[n] += setelt(ll[2]) * setelt(rl[1]) * (τ * B1 + (τ^2 / 2) * anticomm(B1, D))
        # U_t[n] += setelt(ll[2]) * setelt(rl[2]) * ( (τ / 2) * anticomm(B1,C1) )
        # U_t[n] += setelt(ll[2]) * setelt(rl[3]) * ( (τ / 2) * anticomm(B1,C2) )

        # # third row
        # U_t[n] += setelt(ll[3]) * setelt(rl[1]) * (τ * B2 + (τ^2 / 2) * anticomm(B2, D))
        # U_t[n] += setelt(ll[3]) * setelt(rl[2]) * ( (τ / 2) * anticomm(B2, C1) )
        # U_t[n] += setelt(ll[3]) * setelt(rl[3]) * ( (τ / 2) * anticomm(B2, C2) )

    end


    L = ITensor(linkindices[1])
    L[1] = 1.0

    R = ITensor(dag(linkindices[N+1]))
    R[1] = 1.0

    U_t[1] *= L
    U_t[N] *= R

    return U_t

end



function build_expH_heis_secondorder(sites, dt)

    N = length(sites)

    link_dimension = 7

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_t = MPO(N)


    τ = -1.0im * dt


    for n = 1:N
        # siteindex s
        s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        

        Id = op(sites, "Id", n)

        D =  op(sites, "Z",  n)

        B = [op(sites, "S-", n), op(sites, "S+", n)]
        C = [op(sites, "S+", n), op(sites, "S-", n)]

        # Init ITensor inside MPO
        U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))


        #[1,1]
        U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) + (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1))

        # rest of first row 
        for (iiC, Ci) in enumerate(C)
            # sub-block II
            U_t[n] += setelt(ll[1]) * setelt(rl[1+iiC]) * (Ci + (τ / 2) * braket(Ci, D) + (τ^2 / 6) * braket2(Ci, D)) 

            # sub-block III
            for (jjC, Cj) in enumerate(C)
                U_t[n] += setelt(ll[1]) * setelt(rl[1+iiC*2+jjC]) * (replaceprime(Ci' * Cj, 2, 1) + (τ / 3) * braket(Ci, Cj, D))
            end 
        end

        # rest of first column 
        for (iiB, Bi) in enumerate(B)
            # sub-block IV
            U_t[n] += setelt(ll[1+iiB]) * setelt(rl[1]) * (τ * Bi + (τ^2 / 2) * braket(Bi, D) + (τ^3 / 6) * braket2(Bi, D))
            
            #sub-block VII
            for (jjB, Bj) in enumerate(B)
                U_t[n] += setelt(ll[1+ iiB*2 + jjB]) * setelt(rl[1]) * ((τ^2 / 2) * replaceprime(Bi' * Bj, 2, 1) + (τ^3 / 6) * braket(Bi, Bj, D))
            end

        end

        # block V
        for (iiC, Ci) in enumerate(C)
            for (iiB, Bi) in enumerate(B)
                U_t[n] += setelt(ll[1+iiB]) * setelt(rl[1+iiC]) * ( (τ / 2) * braket(Bi, Ci)  + (τ^2 / 6) * braket(Ci, Bi, D) ) 
            end
        end

        # block VI
        for (iiC, Ci) in enumerate(C)
            for (jjC, Cj) in enumerate(C)
                for (iiB, Bi) in enumerate(B)
                    U_t[n] += setelt(ll[1+iiB]) * setelt(rl[1+iiC*2+jjC]) *  (τ / 3) *  braket(Ci, Cj, Bi)#  * ( (τ / 2) * anticomm(Bi,Ci) )
                end
            end
        end

        # block VIII
        for (iiB, Bi) in enumerate(B)
            for (jjB, Bj) in enumerate(B)
                for (iiC, Ci) in enumerate(C)
                    U_t[n] += setelt(ll[1+iiB*2+jjB]) * setelt(rl[1+iiC]) * (τ^2 / 6) * braket(Bi, Bj, Ci)#  * ( (τ / 2) * anticomm(Bi,Ci) )
                end
            end
        end


    end


    L = ITensor(linkindices[1])
    L[1] = 1.0

    R = ITensor(dag(linkindices[N+1]))
    R[1] = 1.0

    U_t[1] *= L
    U_t[N] *= R

    return U_t

end






dt = 0.1 
tsteps = 30

max_chi = 300
SVD_cutoff = 1e-14

N = 20
sites = siteinds("S=1/2", N)

psi_prod = productMPS(ComplexF64, sites, "+") #"↑")



h_auto = build_H_heis_autompo(sites)
h_man = build_H_heis_manual(sites)


println("<auto> = $(abs(inner(psi_prod',h_auto,psi_prod)))")
println("<man> = $(abs(inner(psi_prod',h_man,psi_prod)))")




psi_tdvp = ITensorTDVP.tdvp(
           h_auto,
           psi_prod,
           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
           nsweeps = tsteps,
           maxdim = max_chi,
           cutoff = SVD_cutoff,
           normalize = true,
           outputlevel=1,
)


psi_tdvp2 = ITensorTDVP.tdvp(
           h_man,
           psi_prod,
           -im * dt; # 'real' time evolution according to U(τ) ≈ exp(τ * H) = exp(-im*dt * H)
           nsweeps = tsteps,
           maxdim = max_chi,
           cutoff = SVD_cutoff,
           normalize = true,
           outputlevel=1,
)


psi_u1 = deepcopy(psi_prod)


exp_h_firsto = build_expH_heis_firstorder(sites,dt)


@time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
  psi_u1[:] = apply(exp_h_firsto, psi_u1; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
  println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u1))")
end


psi_u2 = deepcopy(psi_prod)

exp_h_secondo = build_expH_heis_secondorder(sites,dt)

@time for (nt, t) in enumerate(range(dt, step = dt, length = tsteps))
    psi_u2[:] = apply(exp_h_secondo, psi_u2; normalize = true, cutoff = SVD_cutoff, maxdim = max_chi)
    println("nt=$(nt),\tt=$(t),\tmaxbondim = $(maxlinkdim(psi_u2))")
  end


println("<TDVP(ψ) | Ut1(ψ)> = $(abs(inner(psi_tdvp, psi_u1)))")
println("<TDVP(ψ) | Ut1(ψ)> = $(abs(inner(psi_tdvp2, psi_u1)))")
println("<TDVP(ψ) | Ut2(ψ)> = $(abs(inner(psi_tdvp2, psi_u2)))")
println("<Ut1(ψ) | Ut2(ψ)> = $(abs(inner(psi_u1, psi_u2)))")
