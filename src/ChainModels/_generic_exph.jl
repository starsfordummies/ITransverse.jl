""" !TODO NEED TO CHECK IF IT'S CORRECT
Ghent group recipes for 2nd order MPOS""" 
function build_expH_2o(
    sites,
    AString::String,
    JA::Number,
    BString::String,
    JB::Number,
    CString::String,
    JC::Number,
    DString::String,
    JD::Number,
    t;
    )

    # measure τ in imaginary units for actualy time evolution
    τ = -1.0im * t
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
        A = AString == "0" ? 0.0 * op(sites, "Id", n) : JA * op(sites, AString, n)
        B = JB * op(sites, BString, n)
        C = JC * op(sites, CString, n)
        D = JD * op(sites, DString, n)
        # Init ITensor inside MPO
        U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
        # first row
        U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) + (τ^3 / 6) * replaceprime(D'' * D' * D, 3, 1))
        U_t[n] += setelt(ll[1]) * setelt(rl[2]) * (C + (τ / 2) * braket(C, D) + (τ^2 / 6) * braket2(C, D))
        U_t[n] += setelt(ll[1]) * setelt(rl[3]) * (replaceprime(C' * C, 2, 1) + (τ / 3) * braket2(D, C))
        # second row
        U_t[n] += setelt(ll[2]) * setelt(rl[1]) * (τ * B + (τ^2 / 2) * braket(B, D) + (τ^3 / 6) * braket2(B, D))
        U_t[n] += setelt(ll[2]) * setelt(rl[2]) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) + (τ^2 / 6) * (braket(C, B, D) + braket2(A, D)))
        U_t[n] += setelt(ll[2]) * setelt(rl[3]) * (braket(A, C) + (τ / 3) * (braket(A, C, D) + braket2(B, C)))
        # third row
        U_t[n] += setelt(ll[3]) * setelt(rl[1]) * ((τ^2 / 2) * replaceprime(B' * B, 2, 1) + (τ^3 / 6) * braket2(D, B))
        U_t[n] += setelt(ll[3]) * setelt(rl[2]) * ((τ / 2) * braket(A, B) + (τ^2 / 6) * (braket(A, B, D) + braket2(C, B)))
        U_t[n] += setelt(ll[3]) * setelt(rl[3]) * (replaceprime(A' * A, 2, 1) + (τ / 3) * (braket(A, B, C) + braket2(D, A)))
    end

    # implementing OBC: project out upper row and fist column for right and left boundaries, respectively
    L = ITensor(linkindices[1])
    L[1] = 1.0

    R = ITensor(dag(linkindices[N+1]))
    R[1] = 1.0

    U_t[1] *= L
    U_t[N] *= R

    return U_t
end

function build_expH_1o(
    sites,
    AString::String,
    JA::Number,
    BString::String,
    JB::Number,
    CString::String,
    JC::Number,
    DString::String,
    JD::Number,
    t;
    )
    # measure τ in imaginary units for actualy time evolution
    τ = -1.0im * t
    N = length(sites)
    U_t = MPO(N)

    link_dimension = 2
    # startState = 3
    # endState = 1

    linkindices = [Index(link_dimension, "Link,l=$(n-1)") for n = 1:N+1]

    for n = 1:N
        # siteindex s
        s = sites[n]
        # left link index ll with daggered QN conserving direction (if applicable)
        ll = dag(linkindices[n])
        # right link index rl
        rl = linkindices[n+1]
        Id = op(sites, "Id", n)
        # A is possible exponential decay so test for "0"
        A = AString == "0" ? 0.0 * op(sites, "Id", n) : JA * op(sites, AString, n)
        B = JB * op(sites, BString, n)
        C = JC * op(sites, CString, n)
        D = JD * op(sites, DString, n)
        # Init ITensor inside MPO
        U_t[n] = ITensor(ComplexF64, ll, dag(s), s', dag(rl))
        # first row
        U_t[n] += setelt(ll[1]) * setelt(rl[1]) * (Id + τ * D + (τ^2 / 2) * replaceprime(D' * D, 2, 1) )
        U_t[n] += setelt(ll[1]) * setelt(rl[2]) * (C + (τ / 2) * braket(C, D))
        # second row
        U_t[n] += setelt(ll[2]) * setelt(rl[1]) * (τ * B + (τ^2 / 2) * braket(B, D))
        U_t[n] += setelt(ll[2]) * setelt(rl[2]) * (A + (τ / 2) * (braket(B, C) + braket(A, D)) )
    end

    # location changes with convention of upper or lower triangular Matrix as MPO
    # here we have column-major convention i.e. lower-triangular
    L = ITensor(linkindices[1])
    L[1] = 1.0

    R = ITensor(dag(linkindices[N+1]))
    R[1] = 1.0

    U_t[1] *= L
    U_t[N] *= R

    return U_t
end

