function build_H_ising_parallel_field(sites::Vector{<:Index}, mp::model_params)
    build_H_ising_parallel_field(sites, mp.JXX, mp.hz, mp.λx)
end

function build_H_ising_parallel_field(sites::Vector{<:Index}, JXX::Real, hz::Real, λx::Real)

    # Input operator terms which define a Hamiltonian
    N = length(sites)
    os = OpSum()

    for j in 1:(N - 1)
        os += -JXX, "X", j, "X", j + 1
    end

    for j in 1:N
        os += -hz, "Z", j
    end

    for j in 1:N
        os += -λx, "X", j
    end

    # Convert these terms to an MPO tensor network
    return MPO(os, sites)
end


""" Prescription a la Murg for exp(-i*H*dt) Ising transverse+parallel
Convention H = -( JXX + gzZ + λxX )
"""
function build_expH_ising_parallel_field_murg(
    sites::Vector{<:Index},
    JXX::Real,
    gz::Real,
    λx::Real,
    dt::Number)
    """ Symmetric version of Murg exp(-iHising t) """

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    dt = JXX*dt

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
            U_t[n] = onehot(rl => 1) * sqrt(cos(dt))*I
            U_t[n] += onehot(rl => 2) * sqrt(im*sin(dt))*X
        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] = onehot(ll => 1) * sqrt(cos(dt))*I
            U_t[n] += onehot(ll => 2) * sqrt(im*sin(dt))*X

        else
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))

            U_t[n] = onehot(ll => 1, rl =>1) * cos(dt)*I
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(dt))*sqrt(cos(dt))*X
            U_t[n] += onehot(ll => 2, rl =>2) * im*sin(dt)*I
        end

        Ux = exp(im*λx*dt*op(sites, "X", n))
        Uz2 = exp(0.5*im*gz*dt*op(sites, "Z", n))


        # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
        # everything is symmetric in phys legs here so no need to worry too much
        # (otherwise this is not right, transpositions!) 

        U_t[n] *= Uz2' 
        U_t[n] *= Ux
        U_t[n] *= Uz2
        U_t[n] = replaceprime(U_t[n], 2 => 1)

    end

    return U_t


end

function build_expH_ising_parallel_field_murg(s::Vector{<:Index}, p::model_params)
    
    build_expH_ising_parallel_field_murg(s, p.JXX, p.hz, p.λx, p.dt)

end

function build_expH_ising_parallel_field_murg(p::model_params)
    
    space_sites = siteinds(p.phys_space, 3; conserve_qns = false)
    build_expH_ising_parallel_field_murg(space_sites, p.JXX, p.hz, p.λx, p.dt)

end
