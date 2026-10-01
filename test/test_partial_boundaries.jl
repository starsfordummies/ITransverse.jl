using ITensors, ITensorMPS, ITransverse
using Test
using LinearAlgebra

using ITransverse: boundary_tensor, close_boundary, check_boundary, attach_boundary_bottom!,
    vectorized_identity, folded_tMPS

# Infinite-temperature correlators C(x,t) = 2^{-L} Tr[O_c(t) P_y] on small light cones,
# contracted EXACTLY (no truncation), to check two things added for the double light cone:
#   * `folded_tMPO_ext(...; LR_bottom, n_ext_bottom)`: columns that also start late, with
#     spatial-boundary tensors at their foot (the forward cone of the bottom operator);
#   * `inner=true`: a non-product boundary MPS that stops inside the network.

const PAULI = Dict('I' => ComplexF64[1 0; 0 1], 'X' => ComplexF64[0 1; 1 0],
                   'Y' => ComplexF64[0 -im; im 0], 'Z' => ComplexF64[1 0; 0 -1])
botvec(P) = vec(P) ./ 2                  # infinite-temperature bottom, ρ-like
topvec(P) = vec(transpose(P))            # top operator, as `fold_op`

"""
Contract a list of columns exactly. `cols[k]` is an MPS (edges) or MPO (bulk); time-site
legs are renamed to per-bond indices: on a bulk column an unprimed leg connects to the
right neighbour and a primed one to the left, the left edge connects right and the right
edge connects left.
"""
function contract_columns(cols, ts)
    N = length(cols)
    τof = Dict(noprime(ts[τ]) => τ for τ in eachindex(ts))
    B = [[Index(4, "b$(k)_$(τ)") for τ in eachindex(ts)] for k in 1:N-1]
    # bond legs of a non-product boundary MPS (tag "bdry") follow the same rule
    Bb = Dict{Tuple{Index,Int},Index}()
    bb(i, k) = get!(() -> Index(dim(i), "bb$(k)"), Bb, (noprime(i), k))
    E = ITensor(1.0)
    for (k, col) in enumerate(cols), T in col
        for i in inds(T)
            if hastags(i, "bdry")
                T = replaceind(T, i => (plev(i) == 0 ? bb(i, k) : bb(i, k - 1)))
                continue
            end
            hastags(i, "time_fold") && hastags(i, "Site") || continue
            τ = τof[noprime(i)]
            j = k == 1 ? B[1][τ] : k == N ? B[N-1][τ] : (plev(i) == 0 ? B[k][τ] : B[k-1][τ])
            T = replaceind(T, i => j)
        end
        E *= T
    end
    return scalar(E)
end

"""
Light-cone network for O at column c (site 0) and the bottom `bot(s, lo)` (a boundary for
the column at site s whose first time site is lo). `rhombus=false` is the Murg triangle
(every column starts at time 1); `rhombus=true` starts column s at 1 + max(0, |s-y|-1).
"""
function cone_value(b, t, O, bot, y; rhombus::Bool)
    ts = [Index(4, "Site,n=$(k),time_fold") for k in 1:t]
    hi(s) = t - max(0, abs(s) - 1)
    lo(s) = rhombus ? 1 + max(0, abs(s - y) - 1) : 1
    ss = [s for s in -t-abs(y)-1:t+abs(y)+1 if 1 <= lo(s) <= hi(s)]
    N = length(ss)
    cols = Any[]
    for (k, s) in enumerate(ss)
        tts = ts[lo(s):hi(s)]
        if k == 1 || k == N
            push!(cols, MPS(folded_tMPS(b, tts; LR=k == 1 ? :left : :right, rho0=bot(s))))
            continue
        end
        sl, sr = ss[k-1], ss[k+1]
        # top: sites above the OUTER neighbour (away from c) are edge tensors
        n_top = s < 0 ? hi(s) - hi(sl) : s > 0 ? hi(s) - hi(sr) : 0
        LR = s < 0 ? :left : :right
        # bottom: sites below the outer neighbour (away from y)
        n_bot = s < y ? lo(sl) - lo(s) : s > y ? lo(sr) - lo(s) : 0
        LRb = s < y ? :left : :right
        push!(cols, folded_tMPO_ext(b, tts; LR=n_top > 0 ? LR : nothing, n_ext=max(n_top, 0),
                                    LR_bottom=LRb, n_ext_bottom=max(n_bot, 0), rho0=bot(s),
                                    fold_op=s == 0 ? topvec(PAULI[O]) : nothing))
    end
    return contract_columns(cols, ts)
end

""" Murg triangle whose bottom is `bot(s)` (vector or boundary ITensor); `inner(s)` marks the
columns where a boundary MPS ends inside the network. """
function cone_value_mpo(b, t, O, bot, inner)
    ts = [Index(4, "Site,n=$(k),time_fold") for k in 1:t]
    hi(s) = t - max(0, abs(s) - 1)
    ss = collect(-t:t)
    N = length(ss)
    cols = Any[]
    for (k, s) in enumerate(ss)
        if k == 1 || k == N
            push!(cols, MPS(folded_tMPS(b, ts[1:hi(s)]; LR=k == 1 ? :left : :right, rho0=bot(s))))
            continue
        end
        n_top = s < 0 ? hi(s) - hi(ss[k-1]) : s > 0 ? hi(s) - hi(ss[k+1]) : 0
        push!(cols, folded_tMPO_ext(b, ts[1:hi(s)]; LR=n_top > 0 ? (s < 0 ? :left : :right) : nothing,
                                    n_ext=n_top, rho0=bot(s), inner_bottom=inner(s),
                                    fold_op=s == 0 ? topvec(PAULI[O]) : nothing))
    end
    return contract_columns(cols, ts)
end

@testset "partial / double-light-cone boundaries" begin
    mp = IsingParams(1.0, 1.4, 0.9045)
    tp = tMPOParams(mp; dt=0.1, scheme=Murg(), nbeta=0, init_state=botvec(PAULI['I']))
    b = FoldtMPOBlocks(tp)
    t = 3

    @testset "inner=true is opt-in" begin
        s = Index(3, "bdry")
        ip = Index(4, "Site,rho0")
        il, ir = Index(3, "l"), Index(3, "r")
        A = boundary_tensor(random_itensor(ip, il, ir); phys=ip, left=il, right=ir, bond_ind=s)
        e = [1.0, 0, 0]
        Ar = close_boundary(A, e; side=:right, inner=true)
        @test plev(only(filter(i -> !hastags(i, "Site"), inds(Ar)))) == 1   # keeps s'
        @test plev(only(filter(i -> !hastags(i, "Site"), inds(close_boundary(A, e; side=:right))))) == 0
        @test_throws ErrorException check_boundary(Ar)
        @test check_boundary(Ar; inner=true) === Ar
        oo = folded_tMPO(b, [Index(4, "Site,n=1,time_fold"), Index(4, "Site,n=2,time_fold")])
        hook = Index(4, "hook")
        @test_throws ErrorException attach_boundary_bottom!(deepcopy(oo), Ar, hook)
    end

    @testset "double light cone == triangle (exact)" begin
        bI = botvec(PAULI['I'])
        for (O, P) in (('Z', 'Z'), ('X', 'X'), ('Z', 'X'), ('X', 'Y')), y in (0, -1, 2)
            bot(s) = s == y ? botvec(PAULI[P]) : bI
            vt = cone_value(b, t, O, bot, y; rhombus=false)
            vr = cone_value(b, t, O, bot, y; rhombus=true)
            @test isapprox(vr, vt; atol=1e-12)
        end
    end

    @testset "local operator MPO as an inner boundary == sum of its terms (exact)" begin
        # h_y = -(g Z_y + h X_y + J/2 (X_{y-1}X_y + X_y X_{y+1})) as a bond-dimension-3 MPO on
        # columns y-1, y, y+1 (states: 1 nothing placed, 2 X placed awaiting X, 3 done), every
        # other column the product 𝟙/2. The end columns are INNER columns: closed with e1 on
        # the left and e3 on the right via `close_boundary(...; inner=true)`.
        J, g, h = 1.0, 1.4, 0.9045
        bI = botvec(PAULI['I'])
        for y in (0, 1), O in ('Z', 'X')
            sb = Index(3, "bdry")
            function W(s)
                Z2 = zeros(ComplexF64, 2, 2)
                M = [copy(Z2) for _ in 1:3, _ in 1:3]
                M[1, 1] .+= PAULI['I']; M[3, 3] .+= PAULI['I']
                s == y && (M[1, 3] .+= -g .* PAULI['Z'] .- h .* PAULI['X'])
                s in (y - 1, y) && (M[1, 2] .+= (-J / 2) .* PAULI['X'])
                M[2, 3] .+= PAULI['X']
                return M
            end
            function bt(s)
                ip, il, ir = Index(4, "Site,rho0"), Index(3, "l"), Index(3, "r")
                T = zeros(ComplexF64, 4, 3, 3)
                for a in 1:3, c in 1:3
                    T[:, a, c] .= botvec(W(s)[a, c])
                end
                A = boundary_tensor(ITensor(T, ip, il, ir); phys=ip, left=il, right=ir, bond_ind=sb)
                s == y - 1 && return close_boundary(A, [1.0, 0, 0]; side=:left)
                s == y + 1 && return close_boundary(A, [0, 0, 1.0]; side=:right, inner=true)
                return A
            end
            packed = cone_value_mpo(b, t, O, s -> s in (y - 1, y, y + 1) ? bt(s) : bI,
                                    s -> s in (y - 1, y + 1))
            terms = [(-g, Dict(y => 'Z')), (-h, Dict(y => 'X')),
                     (-J / 2, Dict(y - 1 => 'X', y => 'X')), (-J / 2, Dict(y => 'X', y + 1 => 'X'))]
            ref = sum(co * cone_value(b, t, O, s -> haskey(P, s) ? botvec(PAULI[P[s]]) : bI, y;
                                      rhombus=false) for (co, P) in terms)
            @test abs(ref) > 1e-3
            @test isapprox(packed, ref; atol=1e-12)
        end
    end
end
