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
 (or the Σ ↔ τ rotated form for `mp.coupling == :tau`, see `potts_op_names`)
 and 
```
     001
 τ = 100
     010
```
 """

function H_potts(sites_potts, mp::PottsParams)
    (; JSS, ftau) = mp
    A, Ad, F = potts_op_names(mp)
    N = length(sites_potts)

    os = OpSum()
    for j=1:N-1
        os += -JSS,A,j,Ad,j+1
        os += -JSS,Ad,j,A,j+1
    end

    for j=1:N
        os += -ftau,F, j
    end

    H_potts = MPO(os,sites_potts)

    return H_potts
end


""" Potts H MPO built manually with upper-triangular form """
function H_potts_manual(sites_potts, JJ::Real, ff::Real)
 
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

""" Builds Potts H MPO
using the alternate prescription swapping basically Σ ↔ τ
"""
function H_potts_tausigma(sites_potts, JJ, ff)
 
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



########## U(t) builders ###########

"""
    potts_op_names(mp::PottsParams) -> (A, Adag, F)

Operator names of the Potts chain ``H = -J \\sum_i (A_i A^\\dagger_{i+1} + h.c.) - f \\sum_i F_i``
in the form selected by `mp.coupling`:

- `:sigma` - `A = Σ`, `F = τ + τ†` (the historical convention);
- `:tau`   - `A = τ`, `F = Σ + Σ†`, the same model rotated by an on-site Fourier transform.
  Here the Z3 generator `∏ Σ` is diagonal, so this is the form that can carry Z3 QNs
  (`siteinds("S1_Z3", N; conserve_qns=true)`).
"""
function potts_op_names(mp::PottsParams)
    mp.coupling === :sigma && return ("Σ", "Σdag", "τplusτdag")
    mp.coupling === :tau   && return ("τ", "τdag", "ΣplusΣdag")
    throw(ArgumentError("PottsParams.coupling must be :sigma or :tau, got $(mp.coupling)"))
end

function _check_potts_qns(sites, mp::PottsParams)
    if hasqns(sites) && mp.coupling !== :tau
        error("""
            Potts with QN sites needs coupling=:tau: in the Σ-diagonal basis the Z3 generator
            is ∏ τ, which is not diagonal, so τ + τ† has no definite flux. Use
            PottsParams(J, f; coupling=:tau, phys_site=siteind("S1_Z3"; conserve_qns=true)).""")
    end
end

""" `E·W·E` with the symmetric half-step of the on-site field, `E = exp(i f dt/2 F)` """
function _potts_field_halfsteps!(U::MPO, sites, mp::PottsParams, dt::Number)
    fdt = mp.ftau * dt
    abs(fdt) > 1e-10 || return U
    _, _, F = potts_op_names(mp)
    for n in eachindex(U)
        E = exp(im * fdt / 2 * op(F, sites[n]))
        U[n] = product(E, product(U[n], E))
    end
    return U
end


"""
    expH_potts_murg(sites, mp::PottsParams; dt) -> MPO

exp(-iH dt) a la Murg, bond dimension 3. With `A³ = 1` the bond gate is exactly

    exp(ϵ (A⊗A† + A†⊗A)) = Σ_p g_p A^p ⊗ A^{-p},    ϵ = i J dt,
    g_0 = (2e^{-ϵ} + e^{2ϵ})/3,   g_1 = g_2 = (e^{2ϵ} - e^{-ϵ})/3

and bond state `p` carries `A^p` to the left and `A^{-p}` to the right, i.e. Z3 charge `p`:
with QN sites (`coupling=:tau`) the links carry exactly that. The field is split
symmetrically around the bond gates. `dt` already includes the couplings' units; imaginary
`dt = -iβ` gives exp(-βH).

The single tensor is **not** symmetric under L↔R: `W[a,b] = W[-b,-a]`, the transposition maps
the bond charge `p → -p` (spatial reflection reverses the Z3 current), so the rotated column
satisfies `Tᵀ = C T C` with `C` the charge conjugation `p → -p` on every time site. For
boundary states invariant under charge conjugation (with QNs: the neutral state, `[0,0,1]` for
`coupling=:tau`) `C T C = T`, so the whole column *is* symmetric, `Tᵀ = T`, and the symmetric
(L = Rᵀ) machinery applies; for charged boundaries it does not.
[`expH_potts_symmetric_svd`](@ref) gives tensors that are symmetric one by one (no QNs).
"""
function expH_potts_murg(sites, mp::PottsParams; dt::Number)
    _check_potts_qns(sites, mp)
    A, Ad, _ = potts_op_names(mp)

    ϵ = mp.JSS * dt * im
    fI = (2*exp(-ϵ) + exp(2*ϵ))/3
    fA = (exp(2*ϵ) - exp(-ϵ))/3
    g = (fI, fA, fA)

    N = length(sites)
    links = if hasqns(sites)
        [Index([QN("Z", p, 3) => 1 for p in 0:2], "Link,l=$(n-1)"; dir=ITensors.Out) for n in 1:N+1]
    else
        [Index(3, "Link,l=$(n-1)") for n in 1:N+1]
    end

    U = MPO(N)
    for n in 1:N
        s = sites[n]
        powA = (op("Id", s), op(A, s), op(Ad, s))           # A^p, p = 0,1,2
        lf(p) = sqrt(g[p+1]) * powA[p+1]                    # left factor of bond state p
        rf(p) = sqrt(g[p+1]) * powA[mod(-p, 3)+1]           # right factor: A^{-p}
        ll, rl = dag(links[n]), links[n+1]

        U[n] = if n == 1
            sum(onehot(rl => p+1) * lf(p) for p in 0:2)
        elseif n == N
            sum(onehot(ll => p+1) * rf(p) for p in 0:2)
        else
            sum(onehot(ll => a+1, rl => b+1) * product(rf(a), lf(b)) for a in 0:2, b in 0:2)
        end
    end

    return _potts_field_halfsteps!(U, sites, mp, dt)
end


"""
    expH_potts_symmetric_svd(sites, mp::PottsParams; dt) -> MPO

exp(-iH dt) with each bond gate split by a complex-symmetric SVD, `G = Σ_k A_k ⊗ A_k`, so that
the MPO is symmetric under p↔p' and L↔R (what the symmetric transverse algorithms want).
Bond dimension 3; same operator as [`expH_potts_murg`](@ref).

**No QNs**: the symmetric bond basis is made of `A^p ± A^{-p}` combinations, i.e. it mixes
the Z3 charges `±p`. That is unavoidable: an L↔R symmetric tensor sends the same charge both
ways, so for Z3 it can only carry charge 0. With QN sites use [`expH_potts_murg`](@ref)
(`scheme=Murg()`): its tensors are symmetric only up to charge conjugation, but the column
built from them is symmetric for neutral boundary states.
"""
function expH_potts_symmetric_svd(sites, mp::PottsParams; dt::Number)
    hasqns(sites) && error("""
        expH_potts_symmetric_svd: a transposition-symmetric (L↔R) bond basis mixes the Z3
        charges ±1, so it cannot carry QNs. Use scheme=Murg() (expH_potts_murg) with QN sites.""")
    A, Ad, _ = potts_op_names(mp)

    ϵ = mp.JSS * dt * im
    N = length(sites)
    U = MPO(N)

    right_prev = ITensor()
    for n in 1:N-1
        si, sj = sites[n], sites[n+1]
        G = exp(ϵ * (op(A, si) * op(Ad, sj) + op(Ad, si) * op(A, sj)))

        F = symm_svd(G, (si', dag(si)); cutoff=1e-15)
        sqS = sqrt.(F.S)
        link = settags(F.v, "Link,l=$n")
        left  = replaceind(F.U * sqS, F.v => link)
        right = replaceind(sqS * F.V, F.u => link)

        U[n] = n == 1 ? left : product(right_prev, left)
        right_prev = right
    end
    U[N] = right_prev

    return _potts_field_halfsteps!(U, sites, mp, dt)
end
