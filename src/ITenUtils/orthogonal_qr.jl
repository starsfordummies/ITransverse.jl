# ════════════════════════════════════════════════════════════════════════════
# Complex-orthogonal QR:  A = Q R  with  Qᵀ Q = I  (bilinear, no conjugation)
# ════════════════════════════════════════════════════════════════════════════

"""
    complex_orthogonal_qr(A) -> (Q, R)

Thin QR decomposition of a complex matrix using Householder reflectors built
from the **bilinear** inner product xᵀy (no conjugation), so that:

    Qᵀ Q = I    (complex orthogonal — transpose, NOT conjugate transpose)
    A = Q R      (exact, up to floating-point)

Unlike the standard unitary QR (Q†Q = I), this decomposition does NOT
preserve the Hermitian norm, but it does preserve the symmetric bilinear form.
This is the correct gauge transformation for MPS defined over ℂ with a
symmetric (not Hermitian) inner product.
"""
function complex_orthogonal_qr(A::AbstractMatrix{T}) where {T<:Number}
    m, n   = size(A)
    bond   = min(m, n)
    R      = copy(A)

    # Stash the Householder reflectors so we can build the *thin* Q directly
    # (only `bond` columns) instead of materialising the full m×m accumulator.
    us   = Vector{Vector{T}}(undef, bond)   # reflector vectors (uᵀu ≠ 0 entries)
    utus = Vector{T}(undef, bond)
    keep = falses(bond)                     # was the reflector non-degenerate?

    for k in 1:bond
        x = R[k:m, k]

        # Bilinear "norm":  σ = √(xᵀx),  no conjugation  →  σ ∈ ℂ
        σ = sqrt(sum(xi^2 for xi in x))

        # Sign choice: maximise |σ + x₁| to avoid cancellation
        abs(σ + x[1]) < abs(σ - x[1]) && (σ = -σ)

        # Householder vector
        u      = copy(x)
        u[1]  += σ
        utu    = sum(ui^2 for ui in u)          # uᵀu  (bilinear, no conj)

        # `x` (nearly) bilinear-isotropic (xᵀx ≈ 0): the reflector is ill-conditioned, skip it.
        # Q stays complex-orthogonal and A = QR exact, but this column of R is not reduced.
        if abs(utu) < 1e6 * eps(real(T)) * sum(abs2, u)
            @warn "complex_orthogonal_qr: near-isotropic column $k (|uᵀu|/‖u‖² = $(abs(utu) / sum(abs2, u))), reflector skipped; R is not triangular" maxlog=3
            continue
        end

        # Apply  H = I - 2uuᵀ/uᵀu  from the left to R
        R[k:m, k:n] -= 2 .* u * (transpose(u) * R[k:m, k:n]) ./ utu

        us[k] = u; utus[k] = utu; keep[k] = true
    end

    # Build the thin Q = H₁H₂…H_bond · I[:, 1:bond] by applying the reflectors
    # from the left in reverse order. Cost is O(m²·bond) instead of O(m³).
    Q = Matrix{T}(I, m, bond)
    for k in bond:-1:1
        keep[k] || continue
        u   = us[k]
        utu = utus[k]
        Qk  = @view Q[k:m, :]
        Qk .-= (2 / utu) .* (u * (transpose(u) * Qk))
    end

    # Return thin Q (m×bond) and thin R (bond×n) so that A = Q·R for any shape.
    return Q, R[1:bond, :]
end
