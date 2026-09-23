# ════════════════════════════════════════════════════════════════════════════
# Core site-level gauging primitives
# ════════════════════════════════════════════════════════════════════════════

"""
    _gen_qr(A::ITensor, rinds, cind::Index) -> (Q, R)

Complex-orthogonal thin QR of `A` seen as a matrix `(rinds...) × cind`: `A = Q * R` with
`Q` carrying `rinds` plus a fresh index `new` (same tags/arrow as `cind`) and `Qᵀ Q = 1` in the
bilinear sense, and `R` carrying `(dag(new), cind)`.

With QNs, `A` fused over `rinds` is block-sparse with each row sector tied to exactly one
column sector, so it is a direct sum of independent dense blocks: each is factorised on its own
(`new` gets one sector per block, of dimension `min(rows, cols)`). Row sectors with no matching
column sector carry no information about `cind` and are dropped.
"""
function _gen_qr(A::ITensor, rinds, cind::Index)
    cR = combiner(rinds...; tags="rc")
    ri = combinedind(cR)
    Am = permute(A * cR, ri, cind)

    if !hasqns(A)
        Q, R = complex_orthogonal_qr(Array(Am, ri, cind))
        new  = Index(size(Q, 2), tags(cind))
        return ITensor(Q, ri, new) * dag(cR), ITensor(R, dag(new), cind)
    end

    at  = ITensors.tensor(Am)
    bls = sort!(collect(nzblocks(at)); by = b -> b[2])
    ElT = eltype(at)

    Qs = Matrix{ElT}[]; Rs = Matrix{ElT}[]
    for bl in bls
        Qb, Rb = complex_orthogonal_qr(Matrix(NDTensors.blockview(at, bl)))
        push!(Qs, Qb); push!(Rs, Rb)
    end

    new = Index([space(cind)[bl[2]].first => size(Qs[k], 2) for (k, bl) in enumerate(bls)];
                dir=dir(cind), tags=tags(cind))

    Qt = ITensors.BlockSparseTensor(ElT, [NDTensors.Block(bl[1], k) for (k, bl) in enumerate(bls)],
                                    (ri, new))
    Rt = ITensors.BlockSparseTensor(ElT, [NDTensors.Block(k, bl[2]) for (k, bl) in enumerate(bls)],
                                    (dag(new), cind))
    for (k, bl) in enumerate(bls)
        NDTensors.blockview(Qt, NDTensors.Block(bl[1], k)) .= Qs[k]
        NDTensors.blockview(Rt, NDTensors.Block(k, bl[2])) .= Rs[k]
    end
    return itensor(Qt) * dag(cR), itensor(Rt)
end

"""
    _gen_leftorth!(M::MPS, i::Int)

Left-orthogonalise site `i` with complex-orthogonal QR and pass the R factor to site `i+1`.
Afterwards `Mᵀ M = I` on the fused (left link, site) × right link matrix (plain transpose).
"""
function _gen_leftorth!(M::MPS, i::Int)
    # indices as they sit on `M[i]` (with QNs, `linkind` may return the neighbour's dual copy)
    r = only(commoninds(M[i], M[i+1]))
    l = i == 1 ? nothing : only(commoninds(M[i], M[i-1]))
    s = siteinds(M, i)[1]
    Q, R = _gen_qr(M[i], filter(!isnothing, (l, s)), r)
    M[i]   = Q
    M[i+1] = noprime(R * M[i+1])
end

"""
    _gen_rightorth!(M::MPS, i::Int)

Right-orthogonalise site `i` and pass the R factor to site `i-1`.
Afterwards `M Mᵀ = I` on the left link × fused (site, right link) matrix.
"""
function _gen_rightorth!(M::MPS, i::Int)
    l = only(commoninds(M[i], M[i-1]))
    r = i == length(M) ? nothing : only(commoninds(M[i], M[i+1]))
    s = siteinds(M, i)[1]
    Q, R = _gen_qr(M[i], filter(!isnothing, (s, r)), l)
    M[i]   = Q
    M[i-1] = noprime(R * M[i-1])
end


# ════════════════════════════════════════════════════════════════════════════
# Public API:  gen_orthogonalize! / gen_orthogonalize
# ════════════════════════════════════════════════════════════════════════════

"""
    gen_orthogonalize!(M::MPS, center::Int) -> MPS

Bring the MPS `M` into **generalised mixed-canonical form** centred on site
`center`, using the **complex-orthogonal QR** (Qᵀ Q = I, bilinear, no
conjugation) instead of the standard unitary QR (Q†Q = I).

After the call:
- Sites `1, …, center-1` are **left-orthogonal** in the bilinear sense:
      reshape(A[i], χ_L·d, χ_R)ᵀ · reshape(A[i], χ_L·d, χ_R)  =  I
- Sites `center+1, …, N` are **right-orthogonal** in the bilinear sense:
      reshape(A[i], χ_L, d·χ_R)  ·  reshape(A[i], χ_L, d·χ_R)ᵀ  =  I
- Site `center` carries the full (non-unit) norm of the state.

The **physical state represented by the MPS is unchanged** (no truncation).

# Notes
- This is the analogue of `ITensorMPS.orthogonalize!` but with Qᵀ = Q⁻¹
  instead of Q† = Q⁻¹.
- Useful for MPS with a symmetric (not Hermitian) bilinear inner product,
  e.g. complex-symmetric Hamiltonians, Lindblad operators, or MPS over
  fields that do not admit a positive-definite inner product.
- The gauge is **not** the same as the standard ITensor gauge: overlaps
  `⟨ψ|φ⟩` must be computed with the appropriate bilinear contraction.

# Example
```julia
using ITensors, ITensorMPS

sites = siteinds("S=1/2", 10)
psi   = random_mps(ComplexF64, sites; linkdims=8)

gen_orthogonalize!(psi, 5)   # center on site 5
```
"""
function gen_orthogonalize!(M::MPS, center::Int)
    N = length(M)
    1 ≤ center ≤ N || throw(ArgumentError("center=$center out of range [1,$N]"))

    # Left sweep:  sites 1 → center-1
    for i in 1:(center - 1)
        _gen_leftorth!(M, i)
    end

    # Right sweep: sites N → center+1
    for i in N:-1:(center + 1)
        _gen_rightorth!(M, i)
    end

    # Update ITensorMPS book-keeping
    setleftlim!(M, center - 1)
    setrightlim!(M, center + 1)

    return M
end

"""
    gen_orthogonalize(M::MPS, center::Int) -> MPS

Out-of-place version of `gen_orthogonalize!`: returns a new MPS in
complex-orthogonal mixed-canonical form centred on `center`.
"""
gen_orthogonalize(M::MPS, center::Int) = gen_orthogonalize!(copy(M), center)
