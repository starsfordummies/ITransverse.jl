
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
 """
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
Builds exp(Hpotts) with 2nd order approximation from the Ghent group.
Bond dimension is 7 
"""
function build_expH_potts_2o(sites, 
    J::Real, f::Real,
    dt::Number)

    N = length(sites)

    link_dimension = 7

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    U_t = MPO(N)

    # TODO CHECK SIGN
    τ = -1.0im * dt


    for n = 1:N
        # siteindex s
        s = sites[n]

        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]

        

        Id = op(sites, "Id", n)

        D =  -f * op(sites, "τplusτdag",  n)
        B = -J * [op(sites, "Σ", n), op(sites, "Σdag", n)]
        C = [op(sites, "Σdag", n), op(sites, "Σ", n)]

        # Init ITensor inside MPO
        #U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))


        #[1,1]
        U_t[n] = onehot(ll => 1, rl =>1)  * 
        (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) + (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1))

        # rest of first row 
        for (iiC, Ci) in enumerate(C)
            # sub-block II
            U_t[n] +=  onehot(ll => 1, rl =>1+iiC)* #setelt(ll[1]) * setelt(rl[1+iiC]) 
            (Ci + (τ / 2) * braket(Ci, D) + (τ^2 / 6) * braket2(Ci, D)) 

            # sub-block III
            for (jjC, Cj) in enumerate(C)
                U_t[n] += onehot(ll => 1, rl => 1 + iiC*2 + jjC) * #setelt(ll[1]) * setelt(rl[1+iiC*2+jjC]) * 
                (replaceprime(Ci' * Cj, 2, 1) + (τ / 3) * braket(Ci, Cj, D))
            end 
        end

        # rest of first column 
        for (iiB, Bi) in enumerate(B)
            # sub-block IV
            U_t[n] += onehot(ll => 1 + iiB, rl => 1) * # setelt(ll[1+iiB]) * setelt(rl[1]) * 
            (τ * Bi + (τ^2 / 2) * braket(Bi, D) + (τ^3 / 6) * braket2(Bi, D))
            
            #sub-block VII
            for (jjB, Bj) in enumerate(B)
                U_t[n] += onehot(ll => 1 + iiB*2 + jjB , rl =>1) * # setelt(ll[1+ iiB*2 + jjB]) * setelt(rl[1]) *
                ((τ^2 / 2) * replaceprime(Bi' * Bj, 2, 1) + (τ^3 / 6) * braket(Bi, Bj, D))
            end

        end

        # block V
        for (iiC, Ci) in enumerate(C)
            for (iiB, Bi) in enumerate(B)
                U_t[n] += onehot(ll => 1+iiB, rl =>1+iiC) * # setelt(ll[1+iiB]) * setelt(rl[1+iiC]) * 
                ( (τ / 2) * braket(Bi, Ci)  + (τ^2 / 6) * braket(Ci, Bi, D) ) 
            end
        end

        # block VI
        for (iiC, Ci) in enumerate(C)
            for (jjC, Cj) in enumerate(C)
                for (iiB, Bi) in enumerate(B)
                    U_t[n] += onehot(ll => 1+iiB, rl =>1 + iiC*2 + jjC) * # setelt(ll[1+iiB]) * setelt(rl[1+iiC*2+jjC]) *  
                    (τ / 3) *  braket(Ci, Cj, Bi)#  * ( (τ / 2) * anticomm(Bi,Ci) )
                end
            end
        end

        # block VIII
        for (iiB, Bi) in enumerate(B)
            for (jjB, Bj) in enumerate(B)
                for (iiC, Ci) in enumerate(C)
                    U_t[n] += onehot(ll => 1 + iiB*2 + jjB, rl => 1 + iiC) * # setelt(ll[1+iiB*2+jjB]) * setelt(rl[1+iiC]) *
                    (τ^2 / 6) * braket(Bi, Bj, Ci) #  * ( (τ / 2) * anticomm(Bi,Ci) )
                end
            end
        end

    end


    
    L = onehot(linkindices[1] => 1)
    R = onehot(dag(linkindices[N+1] => 1))


    U_t[1] *= L
    U_t[N] *= R

    return U_t

end


""" Functions needed for the expHpotts a la Murg """
function fsumI_a(x::Number)

    return (2*exp(-x) + exp(2*x))/3.

end

function fsumΣ_a(x::Number)

    return (-exp(-x) + exp(2*x))/3.

end


"""
Builds exp(Hpotts) with the expression a la Murg (sin/cos alike)
Bond dimension is 3
"""
function build_expH_potts_murg(sites, 
    J::Real, fpotts::Real,
    dt::Number)

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
        s = sites[n]

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


