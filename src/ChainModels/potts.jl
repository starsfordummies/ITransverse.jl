
ITensors.op(::OpName"Σ",::SiteType"S=1") =
[exp(2*im*pi/3)     0        0 
 0           exp(4*im*pi/3)  0 
 0                 0         1]

ITensors.op(::OpName"Σdag",::SiteType"S=1") =
[exp(-2*im*pi/3)   0         0 
 0         exp(-4*im*pi/3)   0 
 0           0               1]

ITensors.op(::OpName"ΣplusΣdag",::SiteType"S=1") =
[2*cos(2*pi/3)   0         0 
 0         2*cos(4*pi/3)   0 
 0           0               2]

ITensors.op(::OpName"τ",::SiteType"S=1") =
[0 1 0 
 0 0 1 
 1 0 0]

ITensors.op(::OpName"τdag",::SiteType"S=1") =
[0 0 1 
 1 0 0 
 0 1 0]

ITensors.op(::OpName"τplusτdag",::SiteType"S=1") =
[0 1 1 
 1 0 1 
 1 1 0]

ITensors.state(::StateName"+", ::SiteType"S=1") = [1,1,1]/sqrt(3)


ITensors.state(::StateName"Up", ::SiteType"S1_Z3") = [1,0,0]
ITensors.state(::StateName"Dn", ::SiteType"S1_Z3") = [0,0,1]



function ITensors.space(::SiteType"S1_Z3"; conserve_qns=false)
    if conserve_qns
        return [QN("Z",0,3)=>1, QN("Z",1,3)=>1, QN("Z",2,3)=>1]
    end
    return 3
end


ITensors.op(::OpName"Σ",::SiteType"S1_Z3") =
[exp(2*im*pi/3)     0        0 
 0           exp(4*im*pi/3)  0 
 0                 0         1]

ITensors.op(::OpName"Σdag",::SiteType"S1_Z3") =
[exp(-2*im*pi/3)   0         0 
 0         exp(-4*im*pi/3)   0 
 0           0               1]

ITensors.op(::OpName"ΣplusΣdag",::SiteType"S1_Z3") =
[2*cos(2*pi/3)   0         0 
 0         2*cos(4*pi/3)   0 
 0           0               2]

ITensors.op(::OpName"τ",::SiteType"S1_Z3") =
[0 1 0 
 0 0 1 
 1 0 0]

ITensors.op(::OpName"τdag",::SiteType"S1_Z3") =
[0 0 1 
 1 0 0 
 0 1 0]

ITensors.op(::OpName"τplusτdag",::SiteType"S1_Z3") =
[0 1 1 
 1 0 1 
 1 1 0]

 ITensors.op(::OpName"Sz",::SiteType"S1_Z3") =
[1 0 0 
 0 0 0 
 0 0 -1]


 
 """ Potts H MPO built with autoMPO 
 Convention is H = -J Σ Σdag - J Σdag Σ - f τplusτdag
 with Σ = diag(exp(2πi/3), exp(4πi/3), 1) 
 and 
```
     001
 τ = 100
     010
```
 """
 function build_H_potts(sites_potts, mp::PottsParams)
    build_H_potts(sites_potts, mp.JSS, mp.ftau)
 end

function build_H_potts(sites_potts, JJ::Real, ff::Real)
 
    N = length(sites_potts)

    os = OpSum()
    for j=1:N-1
        os += -JJ,"Σ",j,"Σdag",j+1
        os += -JJ,"Σdag",j,"Σ",j+1
    end

    for j=1:N
        os += -ff,"τplusτdag", j
    end

    H_potts = MPO(os,sites_potts)

    return H_potts
end


""" Potts H MPO built manually with upper-triangular form """
function build_H_potts_manual(sites_potts, JJ::Real, ff::Real)
 
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


        # Upper tri
        # Init ITensor inside MPO
        if n == 1
            # Row vector at the left 
            U_x[n] =  onehot(rl => 1) * I
            U_x[n] += onehot(rl => 2) * Σ
            U_x[n] += onehot(rl => 3) * Σd
            U_x[n] += onehot(rl => 4) * -ff*ττd


        elseif n == N
            #U_x[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_x[n] =  onehot(ll => 1) * -ff*ττd
            U_x[n] += onehot(ll => 2) * -JJ*Σd
            U_x[n] += onehot(ll => 3) * -JJ*Σ
            U_x[n] += onehot(ll => 4) * I

        else

            U_x[n] =  onehot(ll => 1, rl =>1) * I
            U_x[n] += onehot(ll => 1, rl =>2) * Σ
            U_x[n] += onehot(ll => 1, rl =>3) * Σd
            U_x[n] += onehot(ll => 1, rl =>4) * -ff*ττd


            U_x[n] += onehot(ll => 2, rl =>4) * -JJ*Σd
            U_x[n] += onehot(ll => 3, rl =>4) * -JJ*Σ
            U_x[n] += onehot(ll => 4, rl =>4) * I

        end

    end

    return U_x
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

""" Builds Potts H MPO
using the alternate prescription swapping basically Σ ↔ τ
"""
function build_H_potts_tausigma(sites_potts, JJ, ff)
 
    N = length(sites_potts)

    os = OpSum()
    for j=1:N-1
        os += -JJ,"τ",j,"τdag",j+1
        os += -JJ,"τdag",j,"τ",j+1
    end

    for j=1:N
        os += -ff,"ΣplusΣdag", j
    end

    H_potts = MPO(os,sites_potts)

    return H_potts
end






"""
Builds exp(Hpotts) with the expression a la Murg (sin/cos alike)
Bond dimension is 3
"""

function build_expH_potts_murg(sites, 
    J::Real, fpotts::Real,
    dt::Number)

    fsumI_a(x) = (2*exp(-x) + exp(2*x))/3.
    fsumΣ_a(x) = (-exp(-x) + exp(2*x))/3.

    N = length(sites)

    link_dimension = 3

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_t = MPO(N)

    # Here should be implemented exp(ϵ (ΣΣdag + ΣdagΣ))
    # for imag time evol we should have ϵ = -i dt
    # but we have an overall -J sign in the ham, so we should take 

    ϵ = J * 1.0im * dt 

    #fI = fsumI(ϵ, 20)
    #fΣ = fsumΣ(ϵ, 20)


    fI = fsumI_a(ϵ)
    fΣ = fsumΣ_a(ϵ)

    for n = 1:N
        # siteindex s
        #s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        # labels the rows
        ll = dag(linkindices[n])

        # right link index rl - labels the columns
        rl = linkindices[n+1]

        I = op(sites, "Id", n)

        Σ = op(sites, "Σ", n)
        Σd = op(sites, "Σdag", n)


        # Init ITensor inside MPO
        if n == 1
            # Row vector at the left 
            U_t[n] =  onehot(rl => 1) * sqrt(fI) * I
            U_t[n] += onehot(rl => 2) * sqrt(fΣ) * Σ
            U_t[n] += onehot(rl => 3) * sqrt(fΣ) * Σd

        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] =  onehot(ll => 1) * sqrt(fI) * I
            U_t[n] += onehot(ll => 2) * sqrt(fΣ) * Σd
            U_t[n] += onehot(ll => 3) * sqrt(fΣ) * Σ
        else

            U_t[n] =  onehot(ll => 1, rl =>1) * fI * I
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(fI*fΣ) * Σ
            U_t[n] += onehot(ll => 1, rl =>3) * sqrt(fI*fΣ) * Σd

            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(fI*fΣ) * Σd
            U_t[n] += onehot(ll => 2, rl =>2) * fΣ * I
            U_t[n] += onehot(ll => 2, rl =>3) * fΣ * Σ

            U_t[n] += onehot(ll => 3, rl =>1) * sqrt(fI*fΣ) * Σ
            U_t[n] += onehot(ll => 3, rl =>2) * fΣ * Σd
            U_t[n] += onehot(ll => 3, rl =>3) * fΣ * I

        end

        # Mutiply by f-tau part

        if fpotts > 1e-10

            ttdag = op(sites, "τplusτdag",  n)
            expT = exp(ϵ * ttdag * fpotts/2)

            U_t[n] = prime(U_t[n]) * expT
            U_t[n] = noprime(U_t[n] * prime(expT), 2)
        end

    end

    return U_t
end


"""
Builds exp(Hpotts) with the expression a la Murg (sin/cos alike),\\
using the alternate prescription swapping basically Σ ↔ τ
Bond dimension is 3
"""
function build_expH_potts_murg_alt(sites, 
    J::Real, fpotts::Real,
    dt::Number)

    fsumI_a(x) = (2*exp(-x) + exp(2*x))/3.
    fsumΣ_a(x) = (-exp(-x) + exp(2*x))/3.

    N = length(sites)

    link_dimension = 3

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_t = MPO(N)

    # Here should be implemented exp(ϵ (ΣΣdag + ΣdagΣ))
    # for imag time evol we should have ϵ = -i dt
    # but we have an overall -J sign in the ham, so we should take 

    ϵ = J * 1.0im * dt 

    fI = fsumI_a(ϵ)
    fτ = fsumΣ_a(ϵ)

    for n = 1:N
        # siteindex s
        #s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        # labels the rows
        ll = dag(linkindices[n])

        # right link index rl - labels the columns
        rl = linkindices[n+1]

        I = op(sites, "Id", n)

        τ = op(sites, "τ", n)
        τd = op(sites, "τdag", n)

        #@show matrix(τ)
        #@show matrix(τd) 

        # Init ITensor inside MPO
        if n == 1
            # Row vector at the left 
            U_t[n] =  onehot(rl => 1) * sqrt(fI) * I
            U_t[n] += onehot(rl => 2) * sqrt(fτ) * τ
            U_t[n] += onehot(rl => 3) * sqrt(fτ) * τd

        elseif n == N
            #U_t[n] = ITensor(ComplexF64, ll, dag(s), s')
            U_t[n] =  onehot(ll => 1) * sqrt(fI) * I
            U_t[n] += onehot(ll => 2) * sqrt(fτ) * τd
            U_t[n] += onehot(ll => 3) * sqrt(fτ) * τ

        else
            U_t[n] =  onehot(ll => 1, rl =>1) * fI * I
            U_t[n] += onehot(ll => 1, rl =>2) * sqrt(fI*fτ) * τ
            U_t[n] += onehot(ll => 1, rl =>3) * sqrt(fI*fτ) * τd

            U_t[n] += onehot(ll => 2, rl =>1) * sqrt(fI*fτ) * τd
            U_t[n] += onehot(ll => 2, rl =>2) * fτ * I
            U_t[n] += onehot(ll => 2, rl =>3) * fτ * τ

            U_t[n] += onehot(ll => 3, rl =>1) * sqrt(fI*fτ) * τ
            U_t[n] += onehot(ll => 3, rl =>2) * fτ * τd
            U_t[n] += onehot(ll => 3, rl =>3) * fτ * I

        end

        # Mutiply by f-tau part

        if fpotts > 1e-10

            ssdag = op(sites, "ΣplusΣdag",  n)
            expT = exp(ϵ * ssdag * fpotts/2)

            U_t[n] = prime(U_t[n]) * expT
            U_t[n] = noprime(U_t[n] * prime(expT), 2)
        end

    end

    return U_t
end



"""
Builds exp(Hpotts) using Symmetric SVD decomposition,\\
should be symmetric (p<->p') and (L<->R)
Bond dimension is 3
"""
function build_expH_potts_symmetric_svd(in_space_sites, 
    J::Real, fpotts::Real,
    dt::Number)

    # ASSERT NEED SYMMETRY p<->p' OR WE SHOuLD BE MORE CAREFUL

    ϵ = J * 1.0im * dt 

    N = length(in_space_sites)

    U_t = MPO(N)

    uT_open = ITensor()

    for n = 1:N-1 # TODO CHECK THIS

        Σi = op(in_space_sites, "Σ", n)
        Σid = op(in_space_sites, "Σdag", n)

        Σj = op(in_space_sites, "Σ", n+1)
        Σjd = op(in_space_sites, "Σdag", n+1)


        e1 = exp(ϵ*(Σi * Σjd + Σid * Σj))

        c1 = combiner(inds(Σi))
        c2 = combiner(inds(Σj))

        e1c = e1 * c1 * c2

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


    end

    U_t[N] = uT_open

    if fpotts > 1e-10
        for n = 1:N

            ttdag = op(in_space_sites, "τplusτdag",  n)
            expT = exp(ϵ * ttdag * fpotts/2)

            U_t[n] = prime(U_t[n], "Site") * expT
            U_t[n] = noprime(U_t[n] * prime(expT, "Site"), 2)
        end
    end

    #@show U_t

    return U_t

end


# Boilerplate

function build_expH_potts_murg(sites, mp::PottsParams, dt::Number)
    build_expH_potts_murg(sites, mp.JSS, mp.ftau, dt)
end

function build_expH_potts_symmetric_svd(in_space_sites, mp::PottsParams, dt::Number) 
    build_expH_potts_symmetric_svd(in_space_sites, mp.JSS, mp.ftau, dt)
end

function build_expH_potts_symmetric_svd(mp::PottsParams, dt::Number) 
    space_sites = siteinds("S=1", 3; conserve_qns = false)
    build_expH_potts_symmetric_svd(space_sites, mp.JSS, mp.ftau, dt)
end

