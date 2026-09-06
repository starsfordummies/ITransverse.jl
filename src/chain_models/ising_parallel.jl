############ Transverse field Ising ##############
###### Our convention is H = -JXX - gZ - hX 


######## Hamiltonian ########

""" Builds Ising Hamiltonian MPO  H = -Jtwo*XX - gperp*Z - hpar*X """ 
function H_ising(sites::Vector{<:Index}, mp::IsingParams)
    (; Jtwo, gperp, hpar) = mp

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
Convention H = -( Jtwo*XX + gperp*Z + λpar*X ) 
"""
function expH_ising_murg(sites::Vector{<:Index}, mp::IsingParams; dt::Number)
    (; Jtwo, gperp, hpar) = mp

    # The Z2 conserved by `conserve_szparity` is P = prod(Z); XX and Z commute with it, a
    # longitudinal X field does not (Rx mixes the two flux sectors), so no QN MPO exists.
    if hasqns(sites) && !iszero(hpar)
        error("""
            Ising with a longitudinal field (hpar=$(hpar)) does not conserve Sz parity:
            X flips the parity, so exp(-i hpar X dt) has no definite flux. Either set
            hpar=0 or build the sites without `conserve_szparity`.""")
    end

    # For real dt this does REAL time evolution 
    # I should have already taken into account both the - sign in exp(-iHt) 
    # and the overall minus in Ising H= -(JXX+Z)

    Uxx = expXX_murg(sites, Jtwo; dt)

    Ux = MPO([op(s, "Rx", θ=-2*hpar*dt) for s in sites])
    Uzhalf = MPO([op(s, "Rz", θ=-gperp*dt) for s in sites])

    # Multiply in order:  exp(iZ/2)*exp(iX)*exp(iXX)*exp(iZ/2)
    
    U_t = iszero(hpar) ? Uzhalf : applyn(Ux, Uzhalf) 
    U_t = applyn(Uxx, U_t) 
    U_t = applyn(Uzhalf, U_t) 

    return U_t

end


""" Symmetric version (Murg) of exp(+iJtwo*dt*XX ) """
function expXX_murg(sites::Vector{<:Index}, Jtwo::Number; dt::Number, make_expZZ::Bool=false)

    Jdt = Jtwo * dt

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
            U_XX[n] = onehot(rl => 1) * sqrt(cos(Jdt))*I
            U_XX[n] += onehot(rl => 2) * sqrt(im*sin(Jdt))*X
        elseif n == N
            U_XX[n] = onehot(ll => 1) * sqrt(cos(Jdt))*I
            U_XX[n] += onehot(ll => 2) * sqrt(im*sin(Jdt))*X

        else
            U_XX[n]  = onehot(ll => 1, rl =>1) * cos(Jdt)*I
            U_XX[n] += onehot(ll => 1, rl =>2) * sqrt(im*sin(Jdt))*sqrt(cos(Jdt))*X
            U_XX[n] += onehot(ll => 2, rl =>1) * sqrt(im*sin(Jdt))*sqrt(cos(Jdt))*X
            U_XX[n] += onehot(ll => 2, rl =>2) * im*sin(Jdt)*I
        end

    end

    return U_XX

end





function expH_ising_symm_svd(s::Vector{<:Index}, mp::IsingParams; dt::Number)
    (; Jtwo, gperp, hpar) = mp

    w = expH_ising_symm_svd_3site(Jtwo, gperp, hpar; dt)
    wmpo = if length(s) == 3
        replace_siteinds(w, s)
    else
        extend_mpo(s, w)
    end
    return wmpo
end

""" Builds core MPO tensors for 3 sites """ 
function expH_ising_symm_svd_3site(Jtwo::Number, hperp::Number, λpar::Number; dt::Number)

    s = siteinds("S=1/2", 3)

    X1 = op(s, "X", 1)
    X2 = op(s, "X", 2)
    X3 = op(s, "X", 3)

    eps = im*dt
    e12 = exp(eps*X1*X2*Jtwo)
    e23 = exp(eps*X2*X3*Jtwo)

    fac_z = hperp*0.5*eps
    eZ1 = exp(fac_z*op(s,"Z",1))
    eZ2 = exp(fac_z*op(s,"Z",2))
    eZ3 = exp(fac_z*op(s,"Z",3))

    eX1 = exp(eps*λpar*op(s,"X",1))
    eX2 = exp(eps*λpar*op(s,"X",2))
    eX3 = exp(eps*λpar*op(s,"X",3))

    l1, r2 = symm_factorization(e12, inds(X1), cutoff=1e-14)
    l2, r3 = symm_factorization(e23, inds(X2), cutoff=1e-14)

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


"""
    expH_ising_murg_4o(sites, mp; dt) -> MPO

4th-order Ising step: the Suzuki five-fold composition `[p,p,1-4p,p,p]` of the
Murg gate, `p = 1/(4-4^{1/3})`. Genuinely 4th order because `expH_ising_murg` is
an exact palindromic Strang splitting, hence time-self-adjoint.

Bond dimension `2^5 = 32`. Thin wrapper over
`build_Ut(sites, Murg(), mp; dt, compose=:suzuki5)`; see [`compose_steps`](@ref).
"""
expH_ising_murg_4o(sites::Vector{<:Index}, mp::IsingParams; dt::Number) =
    build_Ut(sites, Murg(), mp; dt, compose=:suzuki5)

"""
    expH_ising_murg_yoshida(sites, mp; dt) -> MPO

4th-order Ising step by **Yoshida's triple jump** (Phys. Lett. A 150 (1990) 262):
three Murg factors at `[w, 1-2w, w]` with `w = 1/(2-2^{1/3}) ≈ 1.3512`, so the
middle sub-step is a large backward one, `1-2w ≈ -1.7024`. The weights sum to 1
and the composition is palindromic, so like [`expH_ising_murg_4o`](@ref) it is
genuinely 4th order (measured local slope 5).

Two factors fewer than the five-fold composition, at a ~19× larger error
constant (measured). As an integrator that is a losing trade — matching accuracy
needs `dt` smaller by `19.5^(1/4) ≈ 2.1`, i.e. ~6.3 gate applications per unit
time against 5. What it buys is a **composite of bond dimension `D^3` instead of
`D^5`** (8 instead of 32 for Murg; 216 instead of 7776, and 0.03 s instead of
26 s, for a `Ghent3` base), which is what matters when the composite is an
intermediate to be built and contracted densely rather than applied — the input
to `ti_sym_gate`, above all.

Thin wrapper over `build_Ut(sites, Murg(), mp; dt, compose=:yoshida3)`; see
[`compose_steps`](@ref) for the full comparison.
"""
expH_ising_murg_yoshida(sites::Vector{<:Index}, mp::IsingParams; dt::Number) =
    build_Ut(sites, Murg(), mp; dt, compose=:yoshida3)

""" Old name for [`expH_ising_murg_yoshida`](@ref) — `xo` meant "unidentified
order"; it is Yoshida's triple jump, and it is 4th order. """
const expH_ising_murg_xo = expH_ising_murg_yoshida





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
