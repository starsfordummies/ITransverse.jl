using ITensors, ITensorMPS, ITransverse 
using LinearAlgebra
using ITransverse: symmetrize, mytrunc_eig

function make_complex_symmetric(n::Int, n_degen::Int, degen_val=nothing)
    @assert n_degen <= n "Cannot have more degenerate eigenvalues than matrix size"

    # Build eigenvalue vector: n_degen copies of one value, rest random complex
    λ_degen = isnothing(degen_val) ? randn(ComplexF64) : complex(degen_val)
    λ_rest  = randn(ComplexF64, n - n_degen)
    λ       = vcat(fill(λ_degen, n_degen), λ_rest)

    # Build a random invertible V and transpose-orthonormalize it
    # so that Vᵀ V = I  (bilinear, not sesquilinear)
    V_raw = randn(ComplexF64, n, n)

    # Transpose-orthonormalize via bilinear Gram-Schmidt:
    # replace V†V = I (standard) with Vᵀ V = I
    V = _bilinear_orthonormalize(V_raw)

    M = V * Diagonal(λ) * transpose(V)

    # Symmetrize to kill any floating point asymmetry
    return (M + transpose(M)) / 2, λ, V
end

"""
Orthonormalize columns of V under the bilinear (symmetric) inner product ⟨u,v⟩ = uᵀv,
so that Vᵀ V = I.  This is NOT the same as standard QR (which gives V†V = I).
"""
function _bilinear_orthonormalize(V::Matrix{ComplexF64})
    n, m = size(V)
    Q = copy(V)
    for j in 1:m
        # Normalize column j: uᵀu (bilinear, no conjugate)
        nrm = sqrt(sum(Q[:, j] .* Q[:, j]))
        Q[:, j] ./= nrm
        # Subtract projection from remaining columns
        for k in j+1:m
            proj = sum(Q[:, j] .* Q[:, k])   # qⱼᵀ qₖ, no conjugate
            Q[:, k] .-= proj .* Q[:, j]
        end
    end
    return Q
end



function test_symm_oeig(M::AbstractMatrix; maxdim=nothing, cutoff=nothing, use_absolute_cutoff=nothing, use_relative_cutoff=nothing)

    M = symmetrize(M)
    F, spec = mytrunc_eig(M; maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    #dump(F)
    vals = F.values
    vecs = F.vectors

    Z = transpose(vecs) * vecs

    # TODO this is a hack to enforce sqrt of diagonal matrix even when it's only approx diagonal
    diagz = Diagonal(diag(Z))

    if norm(Z - diagz) < 1e-10
        isq_z = diagz^-0.5
    else
      @warn "not diag"
        isq_z_1 = Z^(-0.5)
        Fz = eigen(Z)
        isq_z = (Fz.vectors * Diagonal(Fz.values .^ -0.5)) / Fz.vectors 
        @assert isq_z_1 ≈ isq_z
    end
    O = vecs*isq_z

    M_rec = O * Diagonal(vals) * transpose(O)

    norm_err = norm(M_rec-M)/norm(M)

    if !isnothing(cutoff)
        if norm_err > max(sqrt(cutoff), 1e-12)
            @warn("Ortho/EIG decomp maybe not accurate, norm error $norm_err (cutoff = $cutoff) sqrt=$(sqrt(cutoff))")
        else
            @debug("Ortho/EIG decomp with norm error $(norm_err) < $(sqrt(cutoff)), [norm = $(norm(M))| normS = $(norm(vals))]")
        end
    else
        @warn "No cutoff given"
    end


    return Eigen(vals, O), spec, norm_err
end

m1 = make_complex_symmetric(100, 7)


test_symm_oeig(m1[1])